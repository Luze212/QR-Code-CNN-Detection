import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau # type: ignore
# Alle benötigten Modelle importieren
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50 # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc

# --- KONFIGURATION ---
BASE_DIR = 'dataset_final_boxes'
ROOT_LOG_DIR = os.path.join('logs', 'Augmentation_Transfer_Test')
ROOT_MODEL_DIR = os.path.join('models', 'Augmentation_Transfer_Test')

IMG_SIZE = (300, 300)
BATCH_SIZE = 32

# Trainings-Dauer pro Szenario
EPOCHS_WARMUP = 5
EPOCHS_FINETUNE = 10

# --- AUSWAHL DER MODELLE ---
MODELS_TO_TEST = [
    ("MobileNetV2", MobileNetV2),
    ("VGG16",       VGG16),
    ("ResNet50",    ResNet50)
]

# --- AUGMENTATION SZENARIEN ---
aug_scenarios = [
    {
        "name": "1_Minimal",
        "desc": "Nur Rescaling (Basislinie)",
        "params": { "rescale": 1./255 }
    },
    {
        "name": "2_Light_Geometry",
        "desc": "Leichte Drehung/Verschiebung",
        "params": {
            "rescale": 1./255,
            "rotation_range": 15, "width_shift_range": 0.1, "height_shift_range": 0.1,
            "zoom_range": 0.1, "horizontal_flip": True, "fill_mode": 'nearest'
        }
    },
    {
        "name": "3_Heavy_Geometry",
        "desc": "Starke Verzerrung (Perspektive)",
        "params": {
            "rescale": 1./255,
            "rotation_range": 45, "shear_range": 0.2, "zoom_range": 0.3,
            "width_shift_range": 0.2, "height_shift_range": 0.2,
            "horizontal_flip": True, "fill_mode": 'nearest'
        }
    },
    {
        "name": "4_Lighting_Only",
        "desc": "Licht & Farbe (Waschanlage)",
        "params": {
            "rescale": 1./255,
            "brightness_range": [0.3, 1.7], "channel_shift_range": 50.0,
            "fill_mode": 'nearest'
        }
    },
    {
        "name": "5_Full_Power",
        "desc": "Kombination Geometry & Lighting",
        "params": {
            "rescale": 1./255,
            "rotation_range": 30, "shear_range": 0.2, "zoom_range": 0.2,
            "brightness_range": [0.4, 1.6], "channel_shift_range": 30.0,
            "horizontal_flip": True, "fill_mode": 'nearest'
        }
    }
]

def build_model(ModelClass):
    """Baut das Modell dynamisch basierend auf der Klasse"""
    # Basis laden
    base_model = ModelClass(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False # Erstmal einfrieren
    
    # Custom Head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(320, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Kompilieren für Phase 1
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model, base_model

def main():
    # Ordner erstellen
    os.makedirs(ROOT_LOG_DIR, exist_ok=True)
    os.makedirs(ROOT_MODEL_DIR, exist_ok=True)
    
    print(f"STARTE MULTI-MODEL AUGMENTATION TEST")
    print(f"   Modelle: {[m[0] for m in MODELS_TO_TEST]}")
    print("="*60)

    # ÄUẞERE SCHLEIFE: Durch Modelle iterieren
    for model_name, ModelClass in MODELS_TO_TEST:
        print(f"\n\n{'#'*40}")
        print(f"🤖 MODEL: {model_name}")
        print(f"{'#'*40}")
        
        # Unterordner für dieses Modell anlegen
        current_log_dir = os.path.join(ROOT_LOG_DIR, model_name)
        current_model_dir = os.path.join(ROOT_MODEL_DIR, model_name)
        os.makedirs(current_log_dir, exist_ok=True)
        os.makedirs(current_model_dir, exist_ok=True)
        
        results = []

        # INNERE SCHLEIFE: Durch Augmentations-Szenarien iterieren
        for scenario in aug_scenarios:
            aug_name = scenario['name']
            print(f"\n{model_name} -> Szenario: {aug_name}")
            
            # 1. Daten Generator
            train_datagen = ImageDataGenerator(**scenario['params'])
            val_datagen = ImageDataGenerator(rescale=1./255)

            train_gen = train_datagen.flow_from_directory(
                os.path.join(BASE_DIR, 'train'),
                target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                class_mode='binary', verbose=0
            )
            val_gen = val_datagen.flow_from_directory(
                os.path.join(BASE_DIR, 'val'),
                target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                class_mode='binary', verbose=0, shuffle=False
            )

            # 2. Modell bauen (Speicher vorher leeren!)
            tf.keras.backend.clear_session()
            gc.collect()
            model, base_model = build_model(ModelClass)

            # Log Pfad
            log_path = os.path.join(current_log_dir, f'{aug_name}.csv')
            csv_logger = CSVLogger(log_path)

            # --- PHASE 1: Warmup ---
            print("   Phase 1: Warmup...")
            hist_warmup = model.fit(
                train_gen, epochs=EPOCHS_WARMUP,
                validation_data=val_gen, verbose=0,
                callbacks=[csv_logger]
            )
            
            # --- PHASE 2: Fine-Tuning ---
            print("   Phase 2: Fine-Tuning...")
            base_model.trainable = True
            
            # Intelligentes Unfreezing:
            # VGG16 hat nur 19 Layer, MobileNetV2 >150.
            # Wir tauen die letzten 20% der Layer auf.
            total_layers = len(base_model.layers)
            unfreeze_count = int(total_layers * 0.2) 
            # Mindestens 5 Layer, maximal 50 Layer
            unfreeze_count = max(5, min(unfreeze_count, 50))
            
            # Alle einfrieren bis auf die letzten X
            for layer in base_model.layers[:-unfreeze_count]:
                layer.trainable = False
                
            print(f"      (Taue {unfreeze_count} von {total_layers} Layern auf)")
                
            # Neu kompilieren (Low LR)
            model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                          loss='binary_crossentropy', metrics=['accuracy'])
            
            callbacks_ft = [
                CSVLogger(log_path, append=True),
                EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
            ]
            
            hist_ft = model.fit(
                train_gen, epochs=EPOCHS_FINETUNE,
                validation_data=val_gen, verbose=1,
                callbacks=callbacks_ft
            )

            # --- Auswertung ---
            acc = hist_warmup.history['accuracy'] + hist_ft.history['accuracy']
            val_acc = hist_warmup.history['val_accuracy'] + hist_ft.history['val_accuracy']
            best_val_acc = max(val_acc)
            loss = hist_warmup.history['loss'] + hist_ft.history['loss']
            val_loss = hist_warmup.history['val_loss'] + hist_ft.history['val_loss']

            results.append({
                'Szenario': aug_name,
                'Val Accuracy': round(best_val_acc, 4),
                'Epochen': len(acc)
            })

            # --- PLOTTING (Standardisiertes Design) ---
            epochs_range = range(len(acc))

            plt.figure(figsize=(14, 6))
            
            # --- Linke Seite: Accuracy ---
            ax1 = plt.subplot(1, 2, 1)
            ax1.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
            ax1.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
            
            # Vertikale Linie für Fine-Tuning Start (Grün gestrichelt)
            ax1.axvline(x=EPOCHS_WARMUP-1, color='green', linestyle='--', label='Start Fine-Tune')
            
            ax1.set_title(f'{model_name} - {aug_name}: Accuracy (Best: {best_val_acc:.2%})', fontsize=14)
            ax1.set_xlabel('Epochen')
            ax1.set_ylabel('Accuracy')
            ax1.legend(loc='lower right')
            ax1.grid(True, which='both', linestyle='--', alpha=0.7)

            # --- Rechte Seite: Loss ---
            ax2 = plt.subplot(1, 2, 2)
            ax2.plot(epochs_range, loss, label='Training Loss', linewidth=2, color='red')
            ax2.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2, color='orange')
            
            # Vertikale Linie auch hier
            ax2.axvline(x=EPOCHS_WARMUP-1, color='green', linestyle='--', label='Start Fine-Tune')
            
            ax2.set_title(f'{model_name} - {aug_name}: Loss', fontsize=14)
            ax2.set_xlabel('Epochen')
            ax2.set_ylabel('Loss')
            ax2.legend(loc='upper right')
            ax2.grid(True, which='both', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(current_log_dir, f'{aug_name}.png'), dpi=300)
            plt.close()
            
            # Modell speichern
            model.save(os.path.join(current_model_dir, f'{aug_name}.keras'))

        # --- ZUSAMMENFASSUNG PRO MODELL ---
        print(f"\nErgebnisse für {model_name}:")
        df = pd.DataFrame(results).sort_values(by='Val Accuracy', ascending=False)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(current_log_dir, 'summary_comparison.csv'), index=False)

    print("\nALLE TESTS ABGESCHLOSSEN.")

if __name__ == "__main__":
    main()