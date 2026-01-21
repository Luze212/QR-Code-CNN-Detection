import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc

# --- KONFIGURATION ---
BASE_DIR = 'dataset_final_boxes'  # Dataset
OUTPUT_DIR = 'logs/Argumentation-Tests' # Speicherordner
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 25

# Parameter aus Tuning-Ergebnis
BEST_LR = 0.00055
BEST_DROPOUT = 0.5
BEST_DENSE = 320
BEST_FILTERS = 32 # Nur Block 1 wurde genutzt

# Ordner erstellen
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- AUGMENTATION SZENARIEN ---
aug_scenarios = [
    {
        "name": "1_Minimal",
        "desc": "Nur Rescaling (Referenzwert)",
        "params": {
            "rescale": 1./255
        }
    },
    {
        "name": "2_Light_Geometry",
        "desc": "Leichte Drehung/Verschiebung",
        "params": {
            "rescale": 1./255,
            "rotation_range": 15,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "zoom_range": 0.1,
            "horizontal_flip": True,
            "fill_mode": 'nearest'
        }
    },
    {
        "name": "3_Heavy_Geometry",
        "desc": "Starke Verzerrung (Perspektive/Schräglage)",
        "params": {
            "rescale": 1./255,
            "rotation_range": 45,
            "shear_range": 0.2,
            "zoom_range": 0.3,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "horizontal_flip": True,
            "fill_mode": 'nearest'
        }
    },
    {
        "name": "4_Lighting_Only",
        "desc": "Licht & Farbe (Waschanlagen-Simulation)",
        "params": {
            "rescale": 1./255,
            "brightness_range": [0.3, 1.7],
            "channel_shift_range": 50.0,
            "fill_mode": 'nearest'
        }
    },
    {
        "name": "5_Full_Power",
        "desc": "Kombination aus Heavy Geometry & Lighting",
        "params": {
            "rescale": 1./255,
            "rotation_range": 30,
            "shear_range": 0.2,
            "zoom_range": 0.2,
            "brightness_range": [0.4, 1.6],
            "channel_shift_range": 30.0,
            "horizontal_flip": True,
            "fill_mode": 'nearest'
        }
    }
]

# --- Aufbau getunetes Modell ---
def build_best_model():
    """Baut exakt das Gewinner-Modell aus dem Tuning nach"""
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        
        # 1 Conv Block
        layers.Conv2D(BEST_FILTERS, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        
        # Dense Layer
        layers.Dense(BEST_DENSE, activation='relu'),
        layers.Dropout(BEST_DROPOUT),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = optimizers.Adam(learning_rate=BEST_LR)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def plot_and_save_history(history, folder, filename_prefix, title_prefix):
    """
    Erstellt einen Plot im standardisierten Design (identisch zu Replot_function).
    Zeigt Accuracy (mit Bestwert) und Loss nebeneinander an.
    """
    # Daten aus dem History-Objekt extrahieren
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    # Bestwert ermitteln für den Titel
    best_val_acc = max(val_acc)

    # Plot erstellen (Gleiche Größe wie dein Referenz-Plot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Linke Seite: Accuracy ---
    ax1.plot(epochs, acc, label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
    # Titel mit Bestwert-Anzeige
    ax1.set_title(f'{title_prefix}: Accuracy (Best: {best_val_acc:.2%})', fontsize=14)
    ax1.set_xlabel('Epochen')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # --- Rechte Seite: Loss ---
    # Hier nutzen wir Rot/Orange für bessere Unterscheidung
    ax2.plot(epochs, loss, label='Training Loss', linewidth=2, color='red')
    ax2.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='orange')
    ax2.set_title(f'{title_prefix}: Loss', fontsize=14)
    ax2.set_xlabel('Epochen')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Layout straffen und speichern
    plt.tight_layout()
    
    # Pfad zusammenbauen
    plot_path = os.path.join(folder, f'{filename_prefix}_plot.png')
    
    # Speichern mit hoher Auflösung (300 DPI)
    plt.savefig(plot_path, dpi=300)
    plt.close()

# --- Haupt-Ausführungslogik ---
def main():
    # Zielordner erstellen
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    print(f"Starte Augmentation-Testreihe mit {len(aug_scenarios)} Szenarien...")
    print(f"Datenquelle: {BASE_DIR}")
    print(f"Speicherort: {OUTPUT_DIR}")

    for scenario in aug_scenarios:
        name = scenario['name']
        print(f"\n--- Teste: {name} ({scenario['desc']}) ---")
        
        # 1. Generatoren erstellen
        train_datagen = ImageDataGenerator(**scenario['params'])
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_gen = train_datagen.flow_from_directory(
            os.path.join(BASE_DIR, 'train'),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=True
        )

        val_gen = val_datagen.flow_from_directory(
            os.path.join(BASE_DIR, 'val'),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )

        # 2. Modell bauen
        tf.keras.backend.clear_session()
        model = build_best_model()
        
        # 3. Training
        log_path = os.path.join(OUTPUT_DIR, f'{name}.csv')
        callbacks = [
            CSVLogger(log_path),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # 4. Ergebnis speichern
        best_val_acc = max(history.history['val_accuracy'])
        results.append({
            'Szenario': name,
            'Beschreibung': scenario['desc'],
            'Val Accuracy': round(best_val_acc, 4),
            'Epochen': len(history.history['accuracy'])
        })
        
        # Plotten
        plot_path = os.path.join(OUTPUT_DIR, f'{name}.png')
        plot_and_save_history(history, plot_path, f'{name}_tuned_plot.png', name)
        
        del model
        gc.collect()

    # --- FAZIT ---
    print("\n" + "="*60)
    print("ERGEBNISSE AUGMENTATION")
    print("="*60)
    
    df = pd.DataFrame(results).sort_values(by='Val Accuracy', ascending=False)
    print(df.to_string(index=False))
    
    # Tabelle im Zielordner speichern
    csv_path = os.path.join(OUTPUT_DIR, 'final_augmentation_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nErgebnisse gespeichert in '{csv_path}'")

if __name__ == "__main__":
    main()