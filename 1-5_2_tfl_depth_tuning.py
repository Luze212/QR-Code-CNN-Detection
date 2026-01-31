import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping # type: ignore
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50 # type: ignore
import matplotlib.pyplot as plt
import os
import shutil

# ==========================================
# --- 1. INPUTS AUS PHASE 1 ---
# ==========================================

# Welches Modell optimieren wir gerade?
CHOSEN_MODEL = "MobileNetV2" 
# CHOSEN_MODEL = "VGG16" 
# CHOSEN_MODEL = "ResNet50"

# Trage hier die Ergebnisse aus Skript 1 ein:
TUNED_DENSE_UNITS = 384    
TUNED_DROPOUT = 0.2         
TUNED_LR_PHASE1 = 0.001     
TUNED_OPTIMIZER = 'adam'    

# ==========================================
# --- 2. AUGMENTATION (gleich head_tuning) ---
# ==========================================
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
# --- 3. KONFIGURATION ---
# ==========================================
BASE_LOGS_DIR = f'logs/Phase2_Optimization/{CHOSEN_MODEL}'
DATASET_DIR = 'dataset_final_boxes'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Automatische Definition der Test-Bereiche
if CHOSEN_MODEL == "VGG16":
    # VGG16 (19 Layer): Nur flach testen!
    TEST_UNFREEZE_LAYERS = [4, 8, 12, 15] 
elif CHOSEN_MODEL == "MobileNetV2":
    # MobileNet (155 Layer): Tief testen möglich
    TEST_UNFREEZE_LAYERS = [20, 50, 80, 110]
elif CHOSEN_MODEL == "ResNet50":
    # ResNet50 (175 Layer): Tief testen möglich
    TEST_UNFREEZE_LAYERS = [20, 60, 120, 150]

# Lernraten für Phase 2 testen
TEST_LR_PHASE2 = [1e-6, 1e-5, 5e-5, 1e-4]

BASE_PARAMS = {
    'epochs_phase1': 10, # Warmup
    'epochs_phase2': 15  # Test-Dauer
}

# ==========================================
# --- HELPER & TRAINING ---
# ==========================================
def get_base_model_instance(name, input_shape):
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
    train_gen = train_datagen.flow_from_directory(os.path.join(DATASET_DIR, 'train'), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
    val_gen = val_datagen.flow_from_directory(os.path.join(DATASET_DIR, 'val'), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)
    return train_gen, val_gen

def run_test(unfreeze, lr2, log_subdir):
    # 1. Setup
    t_gen, v_gen = get_generators()
    base_model = get_base_model_instance(CHOSEN_MODEL, IMG_SIZE + (3,))
    base_model.trainable = False
    
    # 2. Modell Bauen (Mit Tuner Werten)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(TUNED_DENSE_UNITS, activation='relu'),
        layers.Dropout(TUNED_DROPOUT),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # --- PHASE 1 (Warmup) ---
    opt1 = get_optimizer(TUNED_OPTIMIZER, TUNED_LR_PHASE1)
    model.compile(optimizer=opt1, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(t_gen, epochs=BASE_PARAMS['epochs_phase1'], validation_data=v_gen, verbose=0)
    
    # --- PHASE 2 (Test) ---
    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - unfreeze)
    for layer in base_model.layers[:freeze_until]: layer.trainable = False
    
    # Phase 2 Optimizer (Adam ist Standard für Fine-Tuning Vergleich)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr2), loss='binary_crossentropy', metrics=['accuracy'])
    
    csv_file = os.path.join(log_subdir, f"log_L{unfreeze}_LR{lr2}.csv")
    
    hist = model.fit(
        t_gen, epochs=BASE_PARAMS['epochs_phase2'], validation_data=v_gen, verbose=0,
        callbacks=[CSVLogger(csv_file), EarlyStopping(patience=5, restore_best_weights=True)]
    )
    return hist.history

def plot_results(results, title, folder):
    plt.figure(figsize=(10, 6))
    for res in results:
        acc = res['history']['val_accuracy']
        plt.plot(acc, label=f"{res['label']} (Max: {max(acc):.2%})")
    plt.title(f'{title} (Phase 2)')
    plt.xlabel('Epochen'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(folder, f'{title}.png')); plt.close()

def main():
    if os.path.exists(BASE_LOGS_DIR): shutil.rmtree(BASE_LOGS_DIR)
    os.makedirs(BASE_LOGS_DIR)
    
    print(f"\n=== PHASE 2 OPTIMIERUNG: {CHOSEN_MODEL} ===")
    
    # --- TEST 1: Auftau-Tiefe ---
    print("\n[1/2] Teste Tiefe (mit LR=1e-5)...")
    res_layers = []
    dir_l = os.path.join(BASE_LOGS_DIR, "Layers")
    os.makedirs(dir_l, exist_ok=True)
    
    for l in TEST_UNFREEZE_LAYERS:
        print(f"   -> {l} Layer...")
        h = run_test(unfreeze=l, lr2=1e-5, log_subdir=dir_l)
        res_layers.append({'label': f"Layers={l}", 'history': h})
    
    plot_results(res_layers, "Unfreeze_Comparison", dir_l)
    best_layer = int(max(res_layers, key=lambda x: max(x['history']['val_accuracy']))['label'].split('=')[1])
    print(f"   >> Gewinner: {best_layer} Layer")

    # --- TEST 2: Lernrate ---
    print(f"\n[2/2] Teste LR (mit Layers={best_layer})...")
    res_lr = []
    dir_lr = os.path.join(BASE_LOGS_DIR, "LR")
    os.makedirs(dir_lr, exist_ok=True)
    
    for lr in TEST_LR_PHASE2:
        print(f"   -> LR {lr}...")
        h = run_test(unfreeze=best_layer, lr2=lr, log_subdir=dir_lr)
        res_lr.append({'label': f"LR={lr}", 'history': h})
        
    plot_results(res_lr, "LR_Comparison", dir_lr)
    best_lr = float(max(res_lr, key=lambda x: max(x['history']['val_accuracy']))['label'].split('=')[1])
    
    print("\n" + "="*30)
    print(f"FINALE EMPFEHLUNG FÜR {CHOSEN_MODEL}")
    print("="*30)
    print(f"Unfreeze Layers: {best_layer}")
    print(f"Phase 2 LR:      {best_lr}")

if __name__ == "__main__":
    main()