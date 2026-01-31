import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50 # type: ignore
import keras_tuner as kt
import os


# ==========================================
# --- 1. KONFIGURATION (HIER ÄNDERN) ---
# ==========================================

# WÄHLE DAS MODELL (nur eins!):
# CHOSEN_MODEL = "MobileNetV2"
CHOSEN_MODEL = "VGG16"
# CHOSEN_MODEL = "ResNet50"

PROJECT_NAME = f"Tuning_Phase1_{CHOSEN_MODEL}" 
LOG_DIR = "logs/Tuning_Phase1"              
DATASET_DIR = 'dataset_final_boxes'

IMG_SIZE = (224, 224) 
BATCH_SIZE = 32
EPOCHS_PER_TRIAL = 12 # Etwas länger, da Augmentation das Lernen verlangsamt

# ==========================================
# --- 2. AUGMENTATION (DEINE FESTEN WERTE) ---
# ==========================================
# Diese Werte sind fixiert aus CNN-Voruntersuchungen.
AUGMENTATION_CONFIG = {
    'rescale': 1./255,
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

def get_base_model(name, input_shape):
    if name == "VGG16": return VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == "MobileNetV2": return MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == "ResNet50": return ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')

def get_generators():
    # Hier wird die Augmentation angewendet
    train_datagen = ImageDataGenerator(**AUGMENTATION_CONFIG)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    return train_gen, val_gen

# ==========================================
# --- 3. TUNER MODELL BUILDER ---
# ==========================================
def build_model(hp):
    input_shape = IMG_SIZE + (3,)
    base_model = get_base_model(CHOSEN_MODEL, input_shape)
    
    # BASIS EINFRIEREN (Wir tunen nur den Kopf!)
    base_model.trainable = False 

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    
    # --- SEARCH SPACE: ARCHITEKTUR ---
    # Wie viele Neuronen brauchen wir für die augmentierten Daten?
    hp_units = hp.Int('dense_units', min_value=128, max_value=512, step=64)
    model.add(layers.Dense(units=hp_units, activation='relu'))
    
    # Wie viel Dropout ist bei dieser Augmentation noch nötig?
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.6, step=0.1)
    model.add(layers.Dropout(rate=hp_dropout))
    
    model.add(layers.Dense(1, activation='sigmoid'))

    # --- SEARCH SPACE: OPTIMIZER ---
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_opt = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    
    if hp_opt == 'adam': opt = optimizers.Adam(learning_rate=hp_lr)
    elif hp_opt == 'sgd': opt = optimizers.SGD(learning_rate=hp_lr)
    elif hp_opt == 'rmsprop': opt = optimizers.RMSprop(learning_rate=hp_lr)
    
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print(f"\n=== STARTE PHASE 1 TUNING (HEAD) FÜR: {CHOSEN_MODEL} ===")
    
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=EPOCHS_PER_TRIAL,
        factor=3,
        directory=LOG_DIR,
        project_name=PROJECT_NAME
    )
    
    train_gen, val_gen = get_generators()
    stop_early = EarlyStopping(monitor='val_loss', patience=3)
    
    tuner.search(train_gen, epochs=EPOCHS_PER_TRIAL, validation_data=val_gen, callbacks=[stop_early])
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"\n=== GEWINNER PHASE 1 ({CHOSEN_MODEL}) ===")
    print(f"Dense Units: {best_hps.get('dense_units')}")
    print(f"Dropout: {best_hps.get('dropout')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    print(f"Optimizer: {best_hps.get('optimizer')}")
    
    # Speichern
    os.makedirs("tuning_results", exist_ok=True)
    with open(f"tuning_results/Phase1_Results_{CHOSEN_MODEL}.txt", "w") as f:
        f.write(f"Model: {CHOSEN_MODEL}\n")
        f.write(f"Dense Units: {best_hps.get('dense_units')}\n")
        f.write(f"Dropout: {best_hps.get('dropout')}\n")
        f.write(f"Learning Rate: {best_hps.get('learning_rate')}\n")
        f.write(f"Optimizer: {best_hps.get('optimizer')}\n")
        f.write(f"Augmentation: Yes (Fixed)\n")

if __name__ == "__main__":
    main()