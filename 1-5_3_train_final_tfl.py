import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50 # type: ignore
import matplotlib.pyplot as plt
import os
import json

# ==========================================
# --- 1. GEWINNER-WERTE EINTRAGEN ---
# ==========================================

# Modell wählen:
CHOSEN_MODEL = "MobileNetV2" 
# CHOSEN_MODEL = "VGG16"
# CHOSEN_MODEL = "ResNet50"

# Werte aus Tuning Phase 1 (Kopf):
TUNED_DENSE_UNITS = 384  
TUNED_DROPOUT = 0.2        
TUNED_LR_PHASE1 = 0.001     
TUNED_OPTIMIZER = 'adam'    

# Werte aus Tuning Phase 2 (Körper):
BEST_UNFREEZE_LAYERS = 80
BEST_LR_PHASE2 = 1e-5    

# ==========================================
# --- 2. KONFIGURATION ---
# ==========================================
PROJECT_NAME = f"Final_Gold_{CHOSEN_MODEL}"
BASE_DIR = "dataset_final_boxes"
LOGS_DIR = f"logs/Final_Production/{CHOSEN_MODEL}"
MODELS_DIR = "models_final"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Epochen (Wie in den Tests)
EPOCHS_PHASE_1 = 10
EPOCHS_PHASE_2 = 20 # Oder 20, wenn du dem finalen Modell mehr Zeit geben willst

# AUGMENTATION (Gesetz! Darf nicht geändert werden)
AUGMENTATION_CONFIG = {
    'rescale': 1./255,
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# ==========================================
# --- 3. HELFER & PLOTTING ---
# ==========================================

# Deine Plotting-Funktion (unverändert)
def plot_and_save_history(history, folder, filename_prefix, title_prefix):
    """
    Erstellt einen Plot im standardisierten Design.
    """
    # Daten aus dem History-Objekt extrahieren
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    best_val_acc = max(val_acc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Linke Seite: Accuracy ---
    ax1.plot(epochs, acc, label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
    # Optional: Linie für Phase 2 Start einzeichnen
    ax1.axvline(x=EPOCHS_PHASE_1-1, color='green', linestyle='--', alpha=0.5, label='Start Fine-Tuning')
    
    ax1.set_title(f'{title_prefix}: Accuracy (Best: {best_val_acc:.2%})', fontsize=14)
    ax1.set_xlabel('Epochen')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # --- Rechte Seite: Loss ---
    ax2.plot(epochs, loss, label='Training Loss', linewidth=2, color='red')
    ax2.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='orange')
    ax2.axvline(x=EPOCHS_PHASE_1-1, color='green', linestyle='--', alpha=0.5, label='Start Fine-Tuning')
    
    ax2.set_title(f'{title_prefix}: Loss', fontsize=14)
    ax2.set_xlabel('Epochen')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(folder, f'{filename_prefix}_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

# Dummy-Klasse, damit deine Plot-Funktion "history.history" aufrufen kann
class CombinedHistory:
    def __init__(self):
        self.history = {}

def get_base_model(name, input_shape):
    if name == "VGG16": return VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == "MobileNetV2": return MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == "ResNet50": return ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')

def get_optimizer(name, lr):
    n = name.lower()
    if n == 'adam': return optimizers.Adam(learning_rate=lr)
    elif n == 'sgd': return optimizers.SGD(learning_rate=lr)
    elif n == 'rmsprop': return optimizers.RMSprop(learning_rate=lr)
    else: return optimizers.Adam(learning_rate=lr)

def get_generators():
    train_datagen = ImageDataGenerator(**AUGMENTATION_CONFIG)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(os.path.join(BASE_DIR, 'train'), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
    val_gen = val_datagen.flow_from_directory(os.path.join(BASE_DIR, 'val'), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)
    return train_gen, val_gen

# ==========================================
# --- 4. HAUPTPROGRAMM ---
# ==========================================
def main():
    # Ordner
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Konfiguration sichern (für die Arbeit/Doku)
    config = {
        "model": CHOSEN_MODEL,
        "phase1_units": TUNED_DENSE_UNITS,
        "phase1_dropout": TUNED_DROPOUT,
        "phase1_lr": TUNED_LR_PHASE1,
        "phase1_opt": TUNED_OPTIMIZER,
        "phase2_layers": BEST_UNFREEZE_LAYERS,
        "phase2_lr": BEST_LR_PHASE2,
        "augmentation": AUGMENTATION_CONFIG
    }
    with open(os.path.join(LOGS_DIR, "final_config.json"), 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"\n=== STARTE FINAL TRAINING: {CHOSEN_MODEL} ===")
    train_gen, val_gen = get_generators()
    
    # --- MODELL BAUEN ---
    print("1. Erstelle Modell...")
    base_model = get_base_model(CHOSEN_MODEL, IMG_SIZE + (3,))
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(TUNED_DENSE_UNITS, activation='relu'),
        layers.Dropout(TUNED_DROPOUT),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # --- PHASE 1: WARMUP ---
    print(f"\n2. Phase 1 (Warmup): {EPOCHS_PHASE_1} Epochen")
    opt1 = get_optimizer(TUNED_OPTIMIZER, TUNED_LR_PHASE1)
    model.compile(optimizer=opt1, loss='binary_crossentropy', metrics=['accuracy'])
    
    cb_p1 = [CSVLogger(os.path.join(LOGS_DIR, "log_phase1.csv"))]
    
    hist1 = model.fit(
        train_gen, 
        epochs=EPOCHS_PHASE_1, 
        validation_data=val_gen,
        callbacks=cb_p1
    )
    
    # --- PHASE 2: FINE-TUNING ---
    print(f"\n3. Phase 2 (Fine-Tuning): {EPOCHS_PHASE_2} Epochen, Unfreeze {BEST_UNFREEZE_LAYERS}")
    
    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - BEST_UNFREEZE_LAYERS)
    for layer in base_model.layers[:freeze_until]: layer.trainable = False
    
    # Wichtig: Immer Adam für Fine-Tuning nehmen (oder den Tuned Optimizer mit kleiner LR)
    # Hier nehmen wir Adam als Standard für Stabilität, es sei denn du willst explizit den Tuned Optimizer
    model.compile(optimizer=optimizers.Adam(learning_rate=BEST_LR_PHASE2), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    cb_p2 = [
        CSVLogger(os.path.join(LOGS_DIR, "log_phase2.csv")),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
        ModelCheckpoint(os.path.join(MODELS_DIR, f"{PROJECT_NAME}_best.keras"), 
                        monitor='val_accuracy', save_best_only=True)
    ]
    
    hist2 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE_1 + EPOCHS_PHASE_2,
        initial_epoch=hist1.epoch[-1] + 1, # Nahtloser Übergang
        validation_data=val_gen,
        callbacks=cb_p2
    )
    
    # --- DATEN ZUSAMMENFÜHREN FÜR PLOT ---
    print("\n4. Erstelle Plots und speichere...")
    
    final_hist = CombinedHistory()
    # Metriken verbinden (Phase 1 Liste + Phase 2 Liste)
    for key in hist1.history.keys():
        final_hist.history[key] = hist1.history[key] + hist2.history[key]
        
    # Plot Funktion aufrufen
    plot_and_save_history(
        final_hist, 
        LOGS_DIR, 
        PROJECT_NAME, 
        f"{CHOSEN_MODEL} (Final)"
    )
    
    # Finales Modell speichern (letzter Stand)
    model.save(os.path.join(MODELS_DIR, f"{PROJECT_NAME}_last.keras"))
    
    print(f"FERTIG! Alles gespeichert in: {LOGS_DIR}")

if __name__ == "__main__":
    main()