import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50 # type: ignore
import matplotlib.pyplot as plt
import os
import time
import gc # Garbage Collector für Speicher-Bereinigung

# --- KONFIGURATION ---
# Pfade (Relativ zum Hauptordner)
BASE_DATA_DIR = 'dataset_final_boxes'
MODELS_OUT_DIR = 'models_tfl'
LOGS_OUT_DIR = os.path.join('logs', 'Transferlearning')

# Hyperparameter für den ersten Testlauf
IMG_SIZE = (300, 300) # Wir bleiben bei der guten Auflösung
BATCH_SIZE = 32
EPOCHS = 20           # Transfer Learning konvergiert meist sehr schnell
LEARNING_RATE = 0.0005 

# --- MODELLE DEFINITION ---
# Liste der zu testenden Architekturen
# Format: (Name für Datei/Log, Keras-Klasse)
transfer_models = [
    ("TL_MobileNetV2", MobileNetV2),
    ("TL_VGG16",       VGG16),
    ("TL_ResNet50",    ResNet50)
]

def get_data_generators():
    """
    Erstellt Generatoren mit moderater Augmentation.
    Nicht zu aggressiv für den Anfang, aber genug für Robustheit.
    """
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

    print(f"Lade Daten aus: {BASE_DATA_DIR}")
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(BASE_DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(BASE_DATA_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_gen, val_gen

def build_transfer_model(model_name, ModelClass):
    """Baut das Modell: Pretrained Base + Custom Head"""
    print(f"   🏗 Baue Modell: {model_name}...")
    
    # 1. Basis laden (Ohne den "Kopf", mit ImageNet Gewichten)
    base_model = ModelClass(
        weights='imagenet', 
        include_top=False, 
        input_shape=IMG_SIZE + (3,)
    )
    
    # 2. Basis einfrieren (WICHTIG für Feature Extraction)
    base_model.trainable = False
    
    # 3. Unseren eigenen Kopf draufsetzen
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), # Modernes Pooling statt Flatten
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy']
    )
    
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

def main():
    # Ordner erstellen
    os.makedirs(MODELS_OUT_DIR, exist_ok=True)
    os.makedirs(LOGS_OUT_DIR, exist_ok=True)
    
    # Daten einmalig initialisieren
    if not os.path.exists(BASE_DATA_DIR):
        print(f"FEHLER: Dataset '{BASE_DATA_DIR}' nicht gefunden.")
        return

    train_gen, val_gen = get_data_generators()

    print("\n" + "="*60)
    print(f"STARTE TRANSFER LEARNING SUITE ({len(transfer_models)} Modelle)")
    print("="*60)

    for model_name, ModelClass in transfer_models:
        print(f"\nStarte Training für: {model_name}")
        
        # Unterordner für Logs dieses Modells
        current_log_dir = os.path.join(LOGS_OUT_DIR, model_name)
        os.makedirs(current_log_dir, exist_ok=True)
        
        # Speicher bereinigen (Wichtig vor jedem großen Modell!)
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Modell bauen
        model = build_transfer_model(model_name, ModelClass)
        
        # Pfade
        model_save_path = os.path.join(MODELS_OUT_DIR, f'{model_name}.keras')
        csv_log_path = os.path.join(current_log_dir, 'history.csv')
        
        callbacks = [
            CSVLogger(csv_log_path),
            ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        
        # Zeitmessung Start
        start_time = time.time()
        
        # Training
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Zeitmessung Ende
        duration = time.time() - start_time
        duration_min = duration / 60
        
        # Auswertung
        best_acc = max(history.history['val_accuracy'])
        plot_and_save_history(history, current_log_dir, f'{model_name}_plot.png', model_name)
        
        # Report schreiben
        with open(os.path.join(current_log_dir, 'report.txt'), 'w') as f:
            f.write(f"REPORT: {model_name}\n")
            f.write("="*30 + "\n")
            f.write(f"Beste Accuracy: {best_acc:.4f}\n")
            f.write(f"Dauer:          {duration_min:.2f} min\n")
            f.write(f"Epochen:        {len(history.history['accuracy'])}\n")
            f.write(f"Basis-Modell:   {ModelClass.__name__}\n")
            f.write(f"Image Size:     {IMG_SIZE}\n")
        
        print(f"{model_name} fertig! (Acc: {best_acc:.2%}, Zeit: {duration_min:.1f} min)")

    print("\n" + "="*60)
    print("ALLE TRANSFER-MODELLE TRAINIERT.")
    print(f"Modelle liegen in: {MODELS_OUT_DIR}")
    print(f"Logs liegen in:    {LOGS_OUT_DIR}")

if __name__ == "__main__":
    main()