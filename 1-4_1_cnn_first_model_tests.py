import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc

# --- KONFIGURATION ---
BASE_DIR = 'dataset_final_boxes'   # Dein Dataset Ordner
LOG_DIR = 'logs/first_argum_test' # Hierhin kommen die CSVs und Plots
MODEL_DIR = 'models'              # Hierhin kommen die .keras Dateien

IMG_SIZE = (256, 256)             # Standardgröße
BATCH_SIZE = 32
EPOCHS = 30

# --- SZENARIEN FÜR DEN VERGLEICH ---
scenarios = [
    {
        "name": "1_Baseline",
        "params": { "lr": 0.001, "dropout": 0.5, "filters": [32, 64, 128] },
        "desc": "Standardeinstellungen"
    },
    {
        "name": "2_High_LR",
        "params": { "lr": 0.005, "dropout": 0.5, "filters": [32, 64, 128] },
        "desc": "Hohe Lernrate (schneller, aber ungenauer?)"
    },
    {
        "name": "3_Small_Network",
        "params": { "lr": 0.001, "dropout": 0.5, "filters": [32] }, # Nur 1 Layer
        "desc": "Kleines Netzwerk (weniger Parameter)"
    },
    {
        "name": "4_Low_Dropout",
        "params": { "lr": 0.001, "dropout": 0.2, "filters": [32, 64, 128] },
        "desc": "Wenig Dropout (Gefahr von Overfitting?)"
    }
]

def build_model(params):
    """Baut das Modell dynamisch anhand der Parameter"""
    model = models.Sequential()
    model.add(layers.Input(shape=IMG_SIZE + (3,)))
    
    # Conv Blöcke dynamisch erstellen
    for f in params['filters']:
        model.add(layers.Conv2D(f, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(params['dropout']))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    optimizer = optimizers.Adam(learning_rate=params['lr'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def main():
    # 1. ORDNER ERSTELLEN
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"🚀 Starte Vergleich von {len(scenarios)} Modellen...")
    results = []

    # Generatoren
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.1, horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    # HIER WAR DER FEHLER: verbose entfernt!
    train_gen = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    val_gen = val_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'val'),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )

    for i, scenario in enumerate(scenarios):
        name = scenario['name']
        params = scenario['params']
        print(f"\n[{i+1}/{len(scenarios)}] Trainiere Modell: {name}")
        print(f"   Einstellungen: LR={params['lr']}, Filters={params['filters']}, Drop={params['dropout']}")
        
        # Sauber machen
        tf.keras.backend.clear_session()
        gc.collect()

        model = build_model(params)

        # Pfade für Output
        log_path = os.path.join(LOG_DIR, f'{name}_log.csv')
        model_path = os.path.join(MODEL_DIR, f'{name}.keras')

        callbacks = [
            CSVLogger(log_path),
            ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=0),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        # Ergebnisse sammeln
        best_acc = max(history.history['val_accuracy'])
        results.append({
            "Modell": name,
            "Beschreibung": scenario['desc'],
            "Val Accuracy": round(best_acc, 4),
            "Epochen": len(history.history['accuracy'])
        })
        
        # Kleinen Plot speichern
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Val')
        plt.title(f'Verlauf: {name}')
        plt.legend()
        plt.savefig(os.path.join(LOG_DIR, f'{name}_plot.png'))
        plt.close()

    # --- ZUSAMMENFASSUNG ---
    print("\n" + "="*50)
    print("🏆 ERGEBNISSE VERGLEICH")
    print("="*50)
    df = pd.DataFrame(results).sort_values(by="Val Accuracy", ascending=False)
    print(df.to_string(index=False))
    
    # Tabelle speichern
    df.to_csv(os.path.join(LOG_DIR, 'comparison_summary.csv'), index=False)
    print(f"\nAlle Ergebnisse gespeichert in: {os.path.abspath(LOG_DIR)}")

if __name__ == "__main__":
    main()