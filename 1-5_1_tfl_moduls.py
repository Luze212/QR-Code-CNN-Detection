import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50 # type: ignore
import matplotlib.pyplot as plt
import time
import gc 
import ssl
import os

# --- FIX: MAC SSL ERROR ---
os.environ['CURL_CA_BUNDLE'] = ''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# --------------------------------------------------

# ==============================
# --- ZENTRALE KONFIGURATION ---
# ==============================

# Pfade
BASE_DATA_DIR = 'dataset_final_boxes'
MODELS_OUT_DIR = 'models_tfl'
LOGS_OUT_DIR = os.path.join('logs', 'Transferlearning')

# 1. Bild-Einstellungen
PARAM_IMG_SIZE = (256, 256) # (224,224) für MobileNetV2 Standard
PARAM_BATCH_SIZE = 32

# 2. Training-Einstellungen
PARAM_EPOCHS = 25           # Maximale Dauer
PARAM_LEARNING_RATE = 0.0001 # Kleiner machen (z.B. 1e-4 statt 5e-4) bei Transfer Learning
PARAM_PATIENCE = 5          # Wie lange warten bei Verschlechterung bevor Stop?

# 3. Architektur-Einstellungen
PARAM_DENSE_UNITS = 256     # Neuronen im Dense Layer
PARAM_DROPOUT_RATE = 0.5    # 0.5 gegen Overfitting

# Modelle 
TRANSFER_MODELS = [
    # ("Modelname", ModelClass), Name anpassbar
    ("TL_MobileNetV2", MobileNetV2),
    ("TL_VGG16",       VGG16),     #sehr groß (>500MB)
    ("TL_ResNet50",    ResNet50)
]

# ============================================================

def get_data_generators():
    """Erstellt Generatoren basierend auf den Parametern oben"""
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
        target_size=PARAM_IMG_SIZE,
        batch_size=PARAM_BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(BASE_DATA_DIR, 'val'),
        target_size=PARAM_IMG_SIZE,
        batch_size=PARAM_BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_gen, val_gen

def build_transfer_model(model_name, ModelClass):
    """Baut das Modell dynamisch mit den Parametern von oben"""
    print(f"   🏗 Baue Modell: {model_name}...")
    
    # 1. Basis laden (ImageNet Gewichte)
    base_model = ModelClass(
        weights='imagenet', 
        include_top=False, 
        input_shape=PARAM_IMG_SIZE + (3,)
    )
    
    # 2. Basis einfrieren
    base_model.trainable = False
    
    # 3. Custom Head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(PARAM_DENSE_UNITS, activation='relu'),
        layers.Dropout(PARAM_DROPOUT_RATE),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(learning_rate=PARAM_LEARNING_RATE),
        metrics=['accuracy']
    )
    
    return model

def plot_and_save_history(history, folder, filename_prefix, title_prefix):
    # Standard Plotting Funktion (Robust)
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
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--')
    
    # Loss
    ax2.plot(epochs, loss, label='Training Loss', linewidth=2, color='red')
    ax2.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='orange')
    ax2.set_title(f'{title_prefix}: Loss', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--')
    
    plt.tight_layout()
    
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        
    plot_path = os.path.join(folder, f'{filename_prefix}_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

def main():
    os.makedirs(MODELS_OUT_DIR, exist_ok=True)
    os.makedirs(LOGS_OUT_DIR, exist_ok=True)
    
    if not os.path.exists(BASE_DATA_DIR):
        print(f"FEHLER: Dataset '{BASE_DATA_DIR}' nicht gefunden.")
        return

    train_gen, val_gen = get_data_generators()

    print("\n" + "="*60)
    print(f"STARTE TRANSFER LEARNING (Konfiguriert: {len(TRANSFER_MODELS)} Modelle)")
    print(f"Parameter: LR={PARAM_LEARNING_RATE}, Dropout={PARAM_DROPOUT_RATE}, ImgSize={PARAM_IMG_SIZE}")
    print("="*60)

    for model_name, ModelClass in TRANSFER_MODELS:
        print(f"\nStarte Training für: {model_name}")
        
        current_log_dir = os.path.join(LOGS_OUT_DIR, model_name)
        os.makedirs(current_log_dir, exist_ok=True)
        
        tf.keras.backend.clear_session()
        gc.collect()
        
        model = build_transfer_model(model_name, ModelClass)
        
        model_save_path = os.path.join(MODELS_OUT_DIR, f'{model_name}.keras')
        csv_log_path = os.path.join(current_log_dir, 'history.csv')
        
        callbacks = [
            CSVLogger(csv_log_path),
            ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=PARAM_PATIENCE, restore_best_weights=True)
        ]
        
        start_time = time.time()
        
        history = model.fit(
            train_gen,
            epochs=PARAM_EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        duration = (time.time() - start_time) / 60
        best_acc = max(history.history['val_accuracy'])
        
        plot_and_save_history(history, current_log_dir, 'training', model_name)
        
        # Report
        with open(os.path.join(current_log_dir, 'report.txt'), 'w') as f:
            f.write(f"Modell: {model_name}\n")
            f.write(f"Accuracy: {best_acc:.4f}\n")
            f.write(f"Params: LR={PARAM_LEARNING_RATE}, Drop={PARAM_DROPOUT_RATE}\n")
        
        print(f"{model_name} fertig. Acc: {best_acc:.2%}")

if __name__ == "__main__":
    main()