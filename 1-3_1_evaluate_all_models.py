import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score # type: ignore
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import seaborn as sns
import pandas as pd
import numpy as np
import os
import shutil

# --- KONFIGURATION ---
BASE_DATA_DIR = 'dataset_final_boxes' # Dataset
MODELS_DIR = 'models' # für models und models_tfl anwendbar
LOGS_BASE_DIR = 'logs/Analyse und Vergleich'

def get_test_generator(input_shape):
    """Erstellt einen Generator passend zur Input-Größe des Modells"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Validierungsordner als Testbasis (oder 'test' falls vorhanden)
    test_dir = os.path.join(BASE_DATA_DIR, 'val') 
    
    gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_shape,
        batch_size=32,
        class_mode='binary',
        shuffle=False,
    )
    return gen

def plot_confusion_matrix(cm, model_name, save_dir):
    """Zeichnet und speichert die Confusion Matrix"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No QR', 'QR'], yticklabels=['No QR', 'QR'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Wahrheit')
    plt.xlabel('Vorhersage')
    
    path = os.path.join(save_dir, f'cm_{model_name}.png')
    plt.savefig(path)
    plt.close()

def plot_prediction_distribution(df, save_dir, threshold):
    """
    Erstellt einen Violin-Plot, der die Verteilung der Vorhersagewahrscheinlichkeiten 
    für jedes Modell zeigt, getrennt nach der wahren Klasse.
    """
    plt.figure(figsize=(14, 8))
    
    # Farben definieren (Blau für No QR, Orange für QR)
    my_palette = {0: "skyblue", 1: "orange"}
    
    # Violinplot
    sns.violinplot(data=df, x='Modell', y='Probability', hue='True Label', 
                   split=True, inner="quart", palette=my_palette, cut=0, linewidth=1.2)
    
    plt.title('Detail-Analyse: Verteilung der Vorhersage-Sicherheit (Split by Class)', fontsize=14)
    plt.xlabel('Modell', fontsize=12)
    plt.ylabel('Vorhersagewahrscheinlichkeit (0.0 = No QR, 1.0 = QR)', fontsize=12)
    
    # Hilfslinie für den gewählten Schwellenwert
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.6, label=f'Threshold {threshold}')
    
    # Y-Achse begrenzen (leicht gepolstert)
    plt.ylim(-0.05, 1.05) 
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # --- LEGENDE ---
    legend_handles = [
        mpatches.Patch(color=my_palette[0], label='Wahrheit: No QR (0)'),
        mpatches.Patch(color=my_palette[1], label='Wahrheit: QR (1)')
    ]
    import matplotlib.lines as mlines
    line_handle = mlines.Line2D([], [], color='red', linestyle='--', label=f'Threshold ({threshold})')
    legend_handles.append(line_handle)

    # ÄNDERUNG: Legende nach unten rechts verschoben
    plt.legend(handles=legend_handles, loc='lower right', frameon=True, shadow=True)
    
    plt.xticks(rotation=45) 
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'prediction_distribution_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Verteilungs-Plot gespeichert: {plot_path}")

def main():
    # 1. User Input
    print("="*40)
    print("MODEL EVALUATIONSTOOL")
    print("="*40)
    test_name = input("Bitte Namen für diesen Testlauf eingeben: ").strip()
    
    if not test_name:
        print("Kein Name eingegeben. Abbruch.")
        return

    try:
        raw_input = input("Bitte Schwellenwert für binäre Klassifikation eingeben (0.0 - 1.0): ").strip()
        PREDICTION_RATE = float(raw_input)
    except ValueError:
        print("Ungültige Zahl. Abbruch.")
        return

    # Ordner erstellen
    save_dir = os.path.join(LOGS_BASE_DIR, test_name)
    if os.path.exists(save_dir):
        choice = input(f"Ordner '{save_dir}' existiert bereits. Überschreiben? (j/n): ")
        if choice.lower() == 'j':
            shutil.rmtree(save_dir)
        else:
            print("Abbruch.")
            return
            
    os.makedirs(save_dir)
    print(f"\nErgebnisse werden gespeichert in: {save_dir}")

    # Modelle finden
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras') or f.endswith('.h5')]
    
    if not model_files:
        print(f"Keine Modelle in '{MODELS_DIR}' gefunden!")
        return

    results = []
    
    # Liste für alle Roh-Vorhersagen aller Modelle (für den Violinplot)
    all_predictions_list = []

    print(f"Starte Evaluation von {len(model_files)} Modellen...\n")

    for i, model_file in enumerate(model_files):
        model_name = model_file.replace('.keras', '').replace('.h5', '')
        print(f"[{i+1}/{len(model_files)}] Prüfe Modell: {model_name} ...")
        
        try:
            # 1. Modell laden
            model_path = os.path.join(MODELS_DIR, model_file)
            model = load_model(model_path)
            
            # 2. Input Shape ermitteln (Dynamisch)
            input_shape = model.input_shape[1:3] 
            
            # 3. Daten Generator
            test_gen = get_test_generator(input_shape)
            
            # 4. Vorhersagen
            # Wir holen hier die rohen Wahrscheinlichkeiten (Floats)
            raw_predictions = model.predict(test_gen, verbose=0).flatten()
            
            y_true = test_gen.classes
            
            # NEU: Daten sammeln für Verteilungs-Plot
            # Wir speichern Modellname, Wahrscheinlichkeit und echtes Label
            df_pred = pd.DataFrame({
                'Modell': model_name,
                'Probability': raw_predictions,
                'True Label': y_true
            })
            all_predictions_list.append(df_pred)
            
            # Binäre Entscheidung basierend auf User-Input
            y_pred = (raw_predictions > PREDICTION_RATE).astype(int)
            
            # 5. Metriken berechnen
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            
            # Details für CSV
            tn, fp, fn, tp = cm.ravel()
            
            # 6. Speichern
            results.append({
                'Modell': model_name,
                'Accuracy': round(acc, 4),
                'F1-Score': round(f1, 4),
                'True Pos (QR gefunden)': tp,
                'False Pos (Fehlalarm)': fp,
                'True Neg (Korrekt ignoriert)': tn,
                'False Neg (QR übersehen)': fn,
                'Input Size': str(input_shape)
            })
            
            # Plot Confusion Matrix
            plot_confusion_matrix(cm, model_name, save_dir)
            
            # Aufräumen
            del model
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"FEHLER bei {model_name}: {e}")

    # --- ABSCHLUSS ---
    print("\n" + "="*40)
    print("FINALE ERGEBNISSE")
    print("="*40)
    
    if results:
        df = pd.DataFrame(results)
        # Sortieren nach Accuracy
        df = df.sort_values(by='Accuracy', ascending=False)
        
        # In Konsole ausgeben
        print(df.to_string(index=False))
        
        # Als CSV speichern
        csv_path = os.path.join(save_dir, 'comparison_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nTabelle gespeichert: {csv_path}")
        
        # Balkendiagramm erstellen
        # Balkendiagramm erstellen (Optimiertes Design)
        plt.figure(figsize=(14, 6))
        
        # Barplot zeichnen
        barplot = sns.barplot(x='Accuracy', y='Modell', data=df, palette='viridis', hue='Modell', legend=False)
        
        # Titel und Achsen anpassen
        plt.title('Modell Vergleich: Accuracy Ranking', fontsize=14)
        plt.xlabel('Accuracy', fontsize=12)
        plt.ylabel('Modell', fontsize=12)
        plt.xlim(0, 1.05) # Etwas Platz rechts lassen für die Zahlen
        
        # Vertikales Gitter für bessere Lesbarkeit
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Die genauen Werte direkt an die Balken schreiben
        for p in barplot.patches:
            width = p.get_width()
            plt.text(
                width + 0.01,       # X-Position (etwas rechts vom Balkenende)
                p.get_y() + p.get_height() / 2, # Y-Position (Mitte des Balkens)
                f'{width:.2%}',     # Text (z.B. "88.35%")
                ha='left',          # Horizontale Ausrichtung
                va='center',        # Vertikale Ausrichtung
                fontsize=11,
                fontweight='bold',
                color='black'
            )
            
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'ranking_plot.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Ranking-Plot gespeichert: {plot_path}")
        
        # --- NEU: Verteilungs-Plots erstellen ---
        if all_predictions_list:
            print("\nErstelle Vorhersage-Verteilungs-Plots (Violin Plot)...")
            full_pred_df = pd.concat(all_predictions_list, ignore_index=True)
            # Wir übergeben hier auch den Threshold, damit die Linie richtig gezeichnet wird
            plot_prediction_distribution(full_pred_df, save_dir, PREDICTION_RATE)
        
    else:
        print("Keine Ergebnisse gesammelt.")

if __name__ == "__main__":
    main()