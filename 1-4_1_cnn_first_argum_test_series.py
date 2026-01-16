import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc # Garbage Collection um Speicher freizugeben

# --- KONFIGURATION ---
BASE_DIR = 'dataset_final_boxes'  # Pfad zum Dataset
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 30 
OUTPUT_DIR = 'log/Argumentation-Tests' # Speicherordner Ergebnisse

# Ordner erstellen
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- VERSUCH-KONFIGURATIONEN ---
experiments = [
    {
        "name": "1_Baseline",
        "learning_rate": 0.001,
        "dropout": 0.5,
        "augmentation": "light",
        "dense_units": 256,
        "filters": [32, 64, 128]
    },
    {
        "name": "2_High_Augmentation",
        "learning_rate": 0.001,
        "dropout": 0.5,
        "augmentation": "heavy",
        "dense_units": 256,
        "filters": [32, 64, 128]
    },
    {
        "name": "3_Low_LR_High_Aug",
        "learning_rate": 0.0001,
        "dropout": 0.5,
        "augmentation": "heavy",
        "dense_units": 256,
        "filters": [32, 64, 128]
    },
    {
        "name": "4_Small_Network",
        "learning_rate": 0.001,
        "dropout": 0.3,
        "augmentation": "light",
        "dense_units": 64,
        "filters": [16, 32, 64]
    }
]

# --- Daten-Generatoren ---
def get_data_generators(aug_level):
    """Erstellt Generatoren basierend auf dem gewählten Augmentation-Level"""
    if aug_level == "heavy":
        # Starke Augmentation (für Versuch 2 & 3)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            brightness_range=[0.4, 1.5],
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        # Leichte Augmentation (für Baseline)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

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
    
    return train_gen, val_gen

# --- Modell-Erstellung ---
def build_model(config):
    """Baut das Modell dynamisch anhand der Config"""
    model = models.Sequential()
    model.add(layers.Input(shape=IMG_SIZE + (3,)))
    
    # Convolutional Blöcke dynamisch erstellen
    for filter_count in config['filters']:
        model.add(layers.Conv2D(filter_count, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(config['dense_units'], activation='relu'))
    model.add(layers.Dropout(config['dropout']))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    optimizer = optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# --- Auswertungsfunktionen ---
def plot_history(history, name):
    """Erstellt und speichert die Trainingskurven"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Accuracy: {name}')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Loss: {name}')
    
    plt.savefig(f'logs/{name}_plot.png')
    plt.close()

# --- Haupt-Ausführungslogik ---
def main():
    # Ordner erstellen
    os.makedirs('models/first argumentation test', exist_ok=True)
    os.makedirs('logs - first argumentation test', exist_ok=True)
    
    results = []

    print(f"🚀 Starte Vergleich von {len(experiments)} Modellen...")

    for i, config in enumerate(experiments):
        name = config['name']
        print(f"\n[{i+1}/{len(experiments)}] Trainiere Modell: {name}")
        print(f"   Einstellungen: LR={config['learning_rate']}, Aug={config['augmentation']}, Filters={config['filters']}")
        
        # 1. Daten laden
        train_gen, val_gen = get_data_generators(config['augmentation'])
        
        # 2. Modell bauen
        tf.keras.backend.clear_session() # Speicher bereinigen
        model = build_model(config)
        
        # 3. Callbacks
        callbacks = [
            CSVLogger(f'logs/{name}_log.csv', append=True),
            ModelCheckpoint(f'models/{name}_best.keras', monitor='val_accuracy', save_best_only=True, verbose=0),
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        ]
        
        # 4. Training
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1 # Fortschrittsbalken
        )
        
        # 5. Auswertung speichern
        plot_history(history, name)
        
        # Werte für Vergleichstabelle
        best_val_acc = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        final_epoch = len(history.history['accuracy'])
        
        results.append({
            'Modell': name,
            'Val Accuracy': round(best_val_acc, 4),
            'Val Loss': round(best_val_loss, 4),
            'Epochen': final_epoch,
            'Augmentation': config['augmentation'],
            'LR': config['learning_rate']
        })
        
        # Speicher aufräumen
        del model
        gc.collect()

    # --- ABSCHLUSS-BERICHT ---
    print("\n" + "="*50)
    print("FINALE ERGEBNISSE")
    print("="*50)
    
    df = pd.DataFrame(results)
    df = df.sort_values(by='Val Accuracy', ascending=False) # Bestes oben
    
    print(df.to_string(index=False))
    
    # Tabelle als CSV speichern
    csv_path = os.path.join(OUTPUT_DIR, 'first_comparison_results.csv')
    df.to_csv(csv_path, index=False)
    print("\nErgebnisse gespeichert in 'first_comparison_results.csv'")

if __name__ == "__main__":
    main()