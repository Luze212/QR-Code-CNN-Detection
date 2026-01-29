import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping # type: ignore
import matplotlib.pyplot as plt
import os
import gc

# --- PFAD KONFIGURATION ---
BASE_DIR = 'dataset_final_boxes' # Dataset
MODELS_DIR = 'models'
LOGS_DIR = 'logs/Argumentation/Precision_2_lightGeom+light_argumentation'
MODUL_NAME = 'Precision_2_lightGeom+light_argumentation'

# Ordner erstellen
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- FINALE HYPERPARAMETER (Hier Werte aus Tuner) ---
TUNING_CONFIG = {
    "start_filters": 64,
    "num_blocks": 4,
    "batch_norm": True,
    "dense_units": 512,
    "dropout": 0.3,
    "learning_rate": 0.00017206382493960582,
    "batch_size": 32,
    "l2_rate": 0.0,           # Falls nicht im Tuner, auf 0.0 lassen
    "img_size": (256, 256)
}

# Training Einstellungen
EPOCHS = 30

# --- DATEN-AUGMENTATION ---
def get_data_generators(batch_size):
    print("Konfiguriere Data Augmentation (Szenario: Light Geometry)...")
    
    # Argumentationsparameter
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        # shear_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',

        # Lightning
        brightness_range=[0.3, 1.7],
        channel_shift_range=50.0,
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),
        target_size=TUNING_CONFIG['img_size'],
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'val'),
        target_size=TUNING_CONFIG['img_size'],
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return train_gen, val_gen

# --- MODELLBAU (DYNAMISCH) ---
def build_dynamic_model(config):
    """
    Baut das Modell automatisch anhand der config-Werte.
    Erstellt Schleifen für die Anzahl der Blöcke.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=config['img_size'] + (3,)))
    
    # L2 Regularisierung setup
    if config.get('l2_rate', 0) > 0:
        reg = regularizers.l2(config['l2_rate'])
    else:
        reg = None

    # --- DYNAMISCHE CONV BLÖCKE ---
    # Hier passiert die Magie: Wir bauen so viele Blöcke, wie 'num_blocks' sagt.
    current_filters = config['start_filters']
    
    for i in range(config['num_blocks']):
        # 1. Convolution
        model.add(layers.Conv2D(
            current_filters, (3, 3), 
            activation='relu',       # Standard Aktivierung
            padding='same',
            kernel_regularizer=reg
        ))
        
        # 2. Pooling
        model.add(layers.MaxPooling2D((2, 2)))
        
        # 3. Batch Normalization (falls im Tuner True)
        if config['batch_norm']:
            model.add(layers.BatchNormalization())
        
        # Filter verdoppeln für den nächsten Block (typische CNN Struktur)
        # Block 1: 32 -> Block 2: 64 -> Block 3: 128 ...
        current_filters *= 2
        
    # --- DENSE LAYERS ---
    model.add(layers.Flatten())
    
    model.add(layers.Dense(
        config['dense_units'], 
        activation='relu',
        kernel_regularizer=reg
    ))
    
    if config['dropout'] > 0:
        model.add(layers.Dropout(config['dropout']))
        
    # Output Layer (Sigmoid für Binär: 0-1)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Kompilieren
    optimizer = optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# --- PLOTTING ---
def plot_and_save_history(history, folder, filename_prefix, title_prefix):
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

# --- MAIN ---
def main():
    print("STARTE TRAINING: Dynamisches Finales CNN")
    print("=======================================")
    print("Konfiguration:")
    for k, v in TUNING_CONFIG.items():
        print(f"  - {k}: {v}")
    
    # 1. Daten laden
    if not os.path.exists(BASE_DIR):
        print(f"FEHLER: Dataset Ordner nicht gefunden: {BASE_DIR}")
        return

    train_gen, val_gen = get_data_generators(TUNING_CONFIG['batch_size'])
    
    # 2. Modell dynamisch bauen & Session clearen
    tf.keras.backend.clear_session()
    gc.collect()
    
    model = build_dynamic_model(TUNING_CONFIG)
    model.summary()
    
    # 3. Callbacks
    model_name = f'{MODUL_NAME}.keras'
    model_save_path = os.path.join(MODELS_DIR, model_name)
    log_save_path = os.path.join(LOGS_DIR, 'training_log.csv')
    
    callbacks = [
        CSVLogger(log_save_path),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ]
    
    # 4. Training
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Speichern
    plot_and_save_history(history, LOGS_DIR, 'final_plot', 'Final Tuned CNN')
    
    print("\nTRAINING ABGESCHLOSSEN")
    print(f"Modell gespeichert: {model_save_path}")
    print(f"Logs gespeichert: {LOGS_DIR}")

if __name__ == "__main__":
    main()