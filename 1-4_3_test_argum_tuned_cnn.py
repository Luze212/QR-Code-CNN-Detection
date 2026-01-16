import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc

# --- KONFIGURATION ---
BASE_DIR = 'dataset_final_boxes'  # Dataset
OUTPUT_DIR = 'log/Argumentation-Tests' # Speicherordner
IMG_SIZE = (300, 300)
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

# --- Haupt-Ausführungslogik ---
def main():
    # Zielordner erstellen
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    print(f"🚀 Starte Augmentation-Testreihe mit {len(aug_scenarios)} Szenarien...")
    print(f"📂 Datenquelle: {BASE_DIR}")
    print(f"💾 Speicherort: {OUTPUT_DIR}")

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
        
        # Plot erstellen und im Zielordner speichern
        plt.figure(figsize=(10, 4))
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.title(f'Verlauf: {name}')
        plt.legend()
        
        plot_path = os.path.join(OUTPUT_DIR, f'{name}.png')
        plt.savefig(plot_path)
        plt.close()
        
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