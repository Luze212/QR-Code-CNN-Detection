import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
import pandas as pd

# --- KONFIGURATION & HYPERPARAMETER ---
PARAM_EPOCHS = 30              # Max. Anzahl Epochen
PARAM_BATCH_SIZE = 32          # Bilder pro Schritt
PARAM_LEARNING_RATE = 0.001    # Lernrate des Optimierers
PARAM_DROPOUT = 0.5            # Dropout Rate (0.5 = 50%)
PARAM_IMG_SIZE = (300, 300)    # Bildgröße (muss zum Datensatz passen)

# --- Name des Versuchs ---
RUN_NAME = f"cnn_custom_lr{PARAM_LEARNING_RATE}_batch{PARAM_BATCH_SIZE}_drop{PARAM_DROPOUT}"
BASE_DIR = 'dataset_test_boxes'     # Dataset Pfad

# --- Aufbau des CNN Modells ---
def create_model(input_shape):
    """Erstellt die CNN Architektur (Aufgabe 3)"""
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=input_shape),
        
        # 1. Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(), # Stabilisiert das Training
        
        # 2. Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # 3. Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten & Dense (Klassifikation)
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(PARAM_DROPOUT), # Verhindert Overfitting
        
        # Output Layer: 1 Neuron für binäre Entscheidung (0=Kein QR, 1=QR)
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# --- Haupt-Ausführungslogik ---
def main():
    print(f"🚀 Starte Training: {RUN_NAME}")
    
    # 1. Augmentation (nur für Training)
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

    print("Lade Daten...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),
        target_size=PARAM_IMG_SIZE,
        batch_size=PARAM_BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'val'),
        target_size=PARAM_IMG_SIZE,
        batch_size=PARAM_BATCH_SIZE,
        class_mode='binary'
    )

    # 2. Modell erstellen und kompilieren
    model = create_model(PARAM_IMG_SIZE + (3,)) # (200, 200, 3)
    
    optimizer = optimizers.Adam(learning_rate=PARAM_LEARNING_RATE)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    model.summary()

    # 3. Callbacks für Logging und Speicherung
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # CSV Logger: Loss/Acc
    csv_logger = CSVLogger(f'logs/{RUN_NAME}.csv', append=True)
    
    # Model Checkpoint: Speichert nur das beste Modell ab
    checkpoint = ModelCheckpoint(
        filepath=f'models/{RUN_NAME}_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=6, # Epochen ohne Verbesserung
        restore_best_weights=True
    )

    # 4. Training starten
    history = model.fit(
        train_generator,
        epochs=PARAM_EPOCHS,
        validation_data=val_generator,
        callbacks=[csv_logger, checkpoint, early_stop]
    )

    # 5. Ergebnisse plotten und speichern
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Accuracy: {RUN_NAME}')
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Loss: {RUN_NAME}')
    
    plt.savefig(f'logs/{RUN_NAME}_plot.png')
    plt.show()
    
    print(f"✅ Training beendet. Bestes Modell gespeichert unter models/{RUN_NAME}_best.keras")

if __name__ == "__main__":
    main()