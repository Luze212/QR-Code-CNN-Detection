import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import shutil

# --- KONFIGURATION ---
BASE_DATA_DIR = 'dataset_final_boxes' # Dataset
MODELS_DIR = 'models' # für models und models_tfl anwendbar
LOGS_BASE_DIR = 'logs'

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
        verbose=0
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

def main():
    # 1. User Input
    print("="*40)
    print("MODEL EVALUATIONSTOOL")
    print("="*40)
    test_name = input("Bitte Namen für diesen Testlauf eingeben: ").strip()
    
    if not test_name:
        print("Kein Name eingegeben. Abbruch.")
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
    print(f"\n📂 Ergebnisse werden gespeichert in: {save_dir}")

    # Modelle finden
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras') or f.endswith('.h5')]
    
    if not model_files:
        print(f"Keine Modelle in '{MODELS_DIR}' gefunden!")
        return

    results = []

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
            predictions = model.predict(test_gen, verbose=0)
            y_pred = (predictions > 0.5).astype(int).flatten()
            y_true = test_gen.classes
            
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
            
            # Plot
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
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Accuracy', y='Modell', data=df, palette='viridis')
        plt.title('Modell Vergleich (Accuracy)')
        plt.xlim(0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ranking_plot.png'))
        print(f"Ranking-Plot gespeichert.")
        
    else:
        print("Keine Ergebnisse gesammelt.")

if __name__ == "__main__":
    main()