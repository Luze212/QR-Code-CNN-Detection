import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
import matplotlib.pyplot as plt
import os

# --- KONFIGURATION ---
BASE_DATA_DIR = 'dataset_final_boxes'
LOGS_DIR = 'logs/tuned_tfl_models'
MODELS_DIR = 'models_tfl_tuned'
IMG_SIZE = (300, 300)
BATCH_SIZE = 32

# Hyperparameter
INITIAL_EPOCHS = 10      # Phase 1: Nur Kopf
FINE_TUNE_EPOCHS = 15    # Phase 2: Alles verfeinern
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

def get_generators():
    # Starke Augmentation für Fine-Tuning (gegen Overfitting beim Auftauen)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.5, 1.5]
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(BASE_DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(BASE_DATA_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    return train_gen, val_gen

def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    train_gen, val_gen = get_generators()

    # --- SCHRITT 1: Basis-Modell laden ---
    print("\nLade MobileNetV2 (ImageNet)...")
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    
    # Erstmal alles einfrieren
    base_model.trainable = False

    # Classifier Kopf
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(320, activation='relu'), # Etwas größer als vorher
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])

    # --- SCHRITT 2: Warm-Up Training (Nur Kopf) ---
    print("\nPHASE 1: Warm-Up Training (Basis eingefroren)...")
    
    history_warmup = model.fit(
        train_gen,
        epochs=INITIAL_EPOCHS,
        validation_data=val_gen,
        verbose=1
    )

    # --- SCHRITT 3: Fine-Tuning (Auftauen) ---
    print("\nPHASE 2: Fine-Tuning (Basis auftauen)...")
    
    # Basis auftauen
    base_model.trainable = True
    
    # WICHTIG: Nicht alles auftauen, sonst zerstören wir zu viel Wissen.
    # MobileNetV2 hat 154 Layer. Wir trainieren nur die letzten ~50.
    print(f"   Anzahl Layer im Base Model: {len(base_model.layers)}")
    fine_tune_at = 100
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # WICHTIG: Modell neu kompilieren mit SEHR KLEINER Lernrate
    # Wir wollen die Gewichte nur "anhauchen", nicht wild ändern.
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), # 0.00001
                  loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks für Phase 2
    callbacks = [
        CSVLogger(os.path.join(LOGS_DIR, 'finetuning_log.csv'), append=True),
        ModelCheckpoint(os.path.join(MODELS_DIR, 'MobileNetV2_FineTuned.keras'), 
                        monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        # Reduziert LR noch weiter, wenn es hakt
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    ]

    # Weiter trainieren
    # initial_epoch ist wichtig, damit der Plot später stimmt (startet bei Epoche 10)
    history_fine = model.fit(
        train_gen,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history_warmup.epoch[-1],
        validation_data=val_gen,
        callbacks=callbacks
    )

   # --- PLOTTING (Kombiniert Phase 1 & 2 im neuen Design) ---
    acc = history_warmup.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history_warmup.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history_warmup.history['loss'] + history_fine.history['loss']
    val_loss = history_warmup.history['val_loss'] + history_fine.history['val_loss']
    
    epochs_range = range(len(acc))
    best_val_acc = max(val_acc)

    plt.figure(figsize=(14, 6))
    
    # --- Linke Seite: Accuracy ---
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
    
    # Vertikale Linie für den Start des Fine-Tunings einzeichnen
    ax1.axvline(x=INITIAL_EPOCHS-1, color='green', linestyle='--', label='Start Fine Tuning')
    
    ax1.set_title(f'Training and Validation Accuracy (Best: {best_val_acc:.2%})', fontsize=14)
    ax1.set_xlabel('Epochen')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)

    # --- Rechte Seite: Loss ---
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs_range, loss, label='Training Loss', linewidth=2, color='red')
    ax2.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2, color='orange')
    
    # Vertikale Linie auch hier einzeichnen
    ax2.axvline(x=INITIAL_EPOCHS-1, color='green', linestyle='--', label='Start Fine Tuning')
    
    ax2.set_title('Training and Validation Loss', fontsize=14)
    ax2.set_xlabel('Epochen')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plot_path = os.path.join(LOGS_DIR, 'finetuning_tfl_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\nFine-Tuning abgeschlossen. Plot gespeichert in {plot_path}")

if __name__ == "__main__":
    main()