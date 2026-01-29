import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc

# --- KONFIGURATION ---
BASE_DIR = 'dataset_final_boxes'  # Dataset Pfad
OUTPUT_DIR = 'logs/Argumentation-Tests_Precision_2' # Speicherordner
IMG_SIZE = (256, 256)
EPOCHS = 35

# --- HIER TUNING WERTE EINTRAGEN ---
TUNING_CONFIG = {
        "l2_rate": 0.001,
        "start_filters": 32,
        "num_blocks": 4,
        "batch_norm": True,
        "dense_units": 512,
        "dropout": 0.3,
        "learning_rate":  0.0005096389594084337,
        "batch_size": 32
}

# Ordner erstellen
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- AUGMENTATION SZENARIEN ---
# (Unverändert, um Vergleichbarkeit zu gewährleisten)
aug_scenarios = [
    {
        "name": "1_Minimal",
        "desc": "Nur Rescaling (Baseline)",
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
        "desc": "Starke Verzerrung",
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
        "desc": "Licht & Farbe",
        "params": {
            "rescale": 1./255,
            "brightness_range": [0.3, 1.7],
            "channel_shift_range": 50.0,
            "fill_mode": 'nearest'
        }
    },
    {
        "name": "5_Full_Power",
        "desc": "Heavy Geometry & Lighting",
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

# --- DYNAMISCHER MODELLBAU ---
def build_tuned_model(config):
    """Baut das Modell dynamisch basierend auf dem Dictionary"""
    model = models.Sequential()
    model.add(layers.Input(shape=IMG_SIZE + (3,)))
    
    # L2 Setup (Falls im Tuning aktiviert)
    if config['l2_rate'] > 0:
        reg = regularizers.l2(config['l2_rate'])
    else:
        reg = None

    # Conv Blöcke dynamisch erstellen
    current_filters = config['start_filters']
    
    for i in range(config['num_blocks']):
        model.add(layers.Conv2D(
            current_filters, (3, 3), 
            activation='relu', 
            padding='same',
            kernel_regularizer=reg
        ))
        model.add(layers.MaxPooling2D((2, 2)))
        
        if config['batch_norm']:
            model.add(layers.BatchNormalization())
            
        current_filters *= 2 # Filter verdoppeln pro Block
        
    model.add(layers.Flatten())
    
    # Dense Layer
    model.add(layers.Dense(
        config['dense_units'], 
        activation='relu',
        kernel_regularizer=reg
    ))
    
    model.add(layers.Dropout(config['dropout']))
    
    # Output
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compiler
    optimizer = optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def plot_and_save_history(history, folder, filename_prefix, title_prefix):
    """Standardisiertes Plotting (14x6, Accuracy & Loss)"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    best_val_acc = max(val_acc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy
    ax1.plot(epochs, acc, label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
    ax1.set_title(f'{title_prefix}: Accuracy (Best: {best_val_acc:.2%})', fontsize=14)
    ax1.set_xlabel('Epochen')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Loss
    ax2.plot(epochs, loss, label='Training Loss', linewidth=2, color='red')
    ax2.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='orange')
    ax2.set_title(f'{title_prefix}: Loss', fontsize=14)
    ax2.set_xlabel('Epochen')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(folder, f'{filename_prefix}_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

# --- HAUPTPROGRAMM ---
def main():
    print(f"🚀 Starte Augmentation-Testreihe mit getunetem Modell")
    print("Konfiguration:")
    for k, v in TUNING_CONFIG.items():
        print(f"  - {k}: {v}")
    
    results = []

    for scenario in aug_scenarios:
        name = scenario['name']
        print(f"\n--- Teste Szenario: {name} ---")
        print(f"Beschreibung: {scenario['desc']}")
        
        # 1. Generatoren (Batch Size aus Config!)
        batch_size = TUNING_CONFIG['batch_size']
        
        train_datagen = ImageDataGenerator(**scenario['params'])
        # Validation immer ohne Augmentation (nur Rescale)
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_gen = train_datagen.flow_from_directory(
            os.path.join(BASE_DIR, 'train'),
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )

        val_gen = val_datagen.flow_from_directory(
            os.path.join(BASE_DIR, 'val'),
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )

        # 2. Modell dynamisch bauen & Session clearen
        tf.keras.backend.clear_session()
        gc.collect() # Wichtig für Mac Speicher
        
        model = build_tuned_model(TUNING_CONFIG)
        
        # 3. Training
        log_path = os.path.join(OUTPUT_DIR, f'{name}.csv')
        callbacks = [
            CSVLogger(log_path),
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        ]
        
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # 4. Daten sammeln
        best_val_acc = max(history.history['val_accuracy'])
        results.append({
            'Szenario': name,
            'Beschreibung': scenario['desc'],
            'Val Accuracy': best_val_acc,
            'Epochen': len(history.history['accuracy'])
        })
        
        # Plotten
        plot_and_save_history(history, OUTPUT_DIR, name, f"{name} ({scenario['desc'][:15]}...)")
        
        # Aufräumen
        del model
        gc.collect()

    # --- AUSWERTUNG ---
    print("\n" + "="*60)
    print("FINALE ERGEBNISSE (Augmentation Tests)")
    print("="*60)
    
    df = pd.DataFrame(results).sort_values(by='Val Accuracy', ascending=False)
    
    # Prozent formatieren für Konsole
    print(df.to_string(formatters={'Val Accuracy': '{:,.2%}'.format}))
    
    # CSV speichern
    csv_path = os.path.join(OUTPUT_DIR, 'final_augmentation_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nVergleichstabelle gespeichert: {csv_path}")
    
    # Ranking Plot
    plt.figure(figsize=(10, 6))
    sns_plot = df.set_index('Szenario')['Val Accuracy'].plot(kind='bar', color='skyblue')
    plt.title("Vergleich der Augmentation-Strategien")
    plt.ylabel("Validation Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ranking_bar_plot.png'))
    print("Ranking-Grafik gespeichert.")

if __name__ == "__main__":
    main()