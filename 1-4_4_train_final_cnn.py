import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping # type: ignore
import matplotlib.pyplot as plt
import os

# --- PFAD KONFIGURATION ---

BASE_DIR = 'dataset_final_boxes' # Dataset
MODELS_DIR = 'models'
LOGS_DIR = 'logs/final_optimized_cnn'

# Ordner erstellen
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- FINALE HYPERPARAMETER ---
PARAM_IMG_SIZE = (256, 256) # Hohe auflösung für bessere Erkennung
PARAM_FILTERS = 32          # Nur 1 Block mit 32 Filtern (Tuning Ergebnis)
PARAM_DENSE = 320           # 320 Neuronen im Dense Layer
PARAM_DROPOUT = 0.5         # 50% Dropout
PARAM_LR = 0.00055          # Optimale Lernrate

# Training
PARAM_BATCH_SIZE = 32
PARAM_EPOCHS = 40           # EarlyStopping bricht ab wenn nötig

# --- DATEN-ARGUMENTATION ---
def get_final_data_generators():
    """
    Kombination aus den Erkenntnissen:
    - Geometrie: Leicht
    - Licht: Aktiviert (da Lighting_Only gut abschnitt und wichtig für Waschanlage ist)
    """
    print("Konfiguriere Data Augmentation (Light Geometry + Lighting)...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        # Leichte Geometrie (Gewinner-Strategie)
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        # Licht-Anpassung (Wichtig für Realitätsnähe in Waschanlage)
        brightness_range=[0.4, 1.6],
        channel_shift_range=30.0,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    print(f"Lade Daten aus: {os.path.abspath(BASE_DIR)}")
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),
        target_size=PARAM_IMG_SIZE,
        batch_size=PARAM_BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'val'),
        target_size=PARAM_IMG_SIZE,
        batch_size=PARAM_BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_gen, val_gen

# --- Modell Definition ---
def build_final_model():
    """Erstellt das optimierte CNN basierend auf Tuning-Ergebnissen"""
    model = models.Sequential([
        layers.Input(shape=PARAM_IMG_SIZE + (3,)),
        
        # 1. Conv Block (1 Block)
        layers.Conv2D(PARAM_FILTERS, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        
        # Dense Block
        layers.Dense(PARAM_DENSE, activation='relu'),
        layers.Dropout(PARAM_DROPOUT),
        
        # Output
        layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = optimizers.Adam(learning_rate=PARAM_LR)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# --- Plot-Funktion ---
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
    print("STARTE TRAINING: Final Optimized CNN")
    print("=======================================")
    
    # 1. Daten laden
    if not os.path.exists(BASE_DIR):
        print(f"FEHLER: Dataset Ordner nicht gefunden: {BASE_DIR}")
        return

    train_gen, val_gen = get_final_data_generators()
    
    # 2. Modell bauen
    model = build_final_model()
    model.summary()
    
    # 3. Callbacks konfigurieren
    model_save_path = os.path.join(MODELS_DIR, 'final_optimized_cnn.keras')
    log_save_path = os.path.join(LOGS_DIR, 'final_optimized_log.csv')
    
    callbacks = [
        CSVLogger(log_save_path),
        # Speichert beste Modell (Val Accuracy)
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ]
    
    # 4. Training starten
    history = model.fit(
        train_gen,
        epochs=PARAM_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Abschluss & Plot
    plot_path = os.path.join(LOGS_DIR, 'final_optimized_plot.png')
    plot_and_save_history(history, plot_path, 'final_optimized_plot.png', 'Final Optimized CNN')
    
    val_acc = max(history.history['val_accuracy'])
    print("\nTRAINING ABGESCHLOSSEN")
    print(f"Beste Validation Accuracy: {val_acc:.4f}")
    print(f"Modell gespeichert in: {model_save_path}")
    print(f"Plot & Log gespeichert in: {LOGS_DIR}")

if __name__ == "__main__":
    main()