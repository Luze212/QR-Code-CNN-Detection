import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16, EfficientNetB0, InceptionV3 # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import shutil

# ==========================================
# --- KONFIGURATION & MODELL-WAHL ---
# ==========================================

# Ordner
BASE_MODELS_DIR = 'models_transfer_opt'
BASE_LOGS_DIR = 'logs_transfer_opt'
DATASET_DIR = 'dataset_final_boxes'
EXPERIMENT_NAME = "TL_Opt"

# Bild
IMG_SIZE = (224, 224)
INPUT_SHAPE = IMG_SIZE + (3,)

# --- MODELL AUSWAHL (Einkommentieren) ---
# CHOSEN_MODEL = "VGG16"
# CHOSEN_MODEL = "ResNet50"
CHOSEN_MODEL = "MobileNetV2" 

def get_base_model(name, input_shape):
    """Lädt die gewählte Architektur ohne Top-Layer."""
    if name == "MobileNetV2": return MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == "ResNet50": return ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == "VGG16": return VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == "EfficientNetB0": return EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == "InceptionV3": return InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')
    else: raise ValueError(f"Modell {name} nicht definiert.")

# --- 1. BASIS HYPERPARAMETER (Tuning-Ergebnisse) ---
BASE_PARAMS = {
    # Phase 1: Warm-Up
    'warmup_epochs': 10,
    'warmup_lr': 0.001,       # Ergebnis aus Head Tuning
    
    # Phase 2: Fine-Tuning
    'finetune_epochs': 15,
    'finetune_lr': 1e-5,      # Ergebnis aus Depth Tuning
    'unfreeze_layers': 80,    # Ergebnis aus Depth Tuning
    
    # Architektur
    'batch_size': 32,
    'optimizer': 'adam',      # Ergebnis aus Head Tuning
    'activation': 'relu',
    'dropout': 0.2,           # Ergebnis Head Tuning
    'dense_units': 384,       # Ergebnis Head Tuning
    'loss': 'binary_crossentropy'
}

# --- 2. EINZEL-PARAMETER TESTS ---
SINGLE_PARAM_TESTS = {
    'finetune_lr': [1e-6, 1e-4],      # Basis:
    'unfreeze_layers': [10, 50],      # Basis:
    'dropout': [0.3, 0.7],            # Basis:
    'dense_units': [128, 512],        # Basis:
    'batch_size': [16, 64],           # Basis:
    'optimizer': ['rmsprop', 'sgd']   # Basis:
}

# --- 3. KOMBINATIONS-TESTS ---
COMBINATION_GROUPS = {
    'Combo_Deep_Dive': { # Tiefes Nachtrainieren
        'Increase (+)': { 
            'unfreeze_layers': 60,
            'finetune_lr': 5e-5, # Etwas höher da mehr Layer
            'dropout': 0.6
        },
        'Decrease (-)': { # Oberflächliches Anpassen
            'unfreeze_layers': 10,
            'finetune_lr': 1e-6,
            'dropout': 0.3
        }
    },
    'Combo_Capacity': { # Größe des Classifiers
        'Increase (+)': { 
            'dense_units': 512,
            'dropout': 0.6
        },
        'Decrease (-)': { 
            'dense_units': 64,
            'dropout': 0.2
        }
    }
}

# ==========================================
# --- DATA GENERATOR ---
# ==========================================
def get_generators(batch_size):
    # Starke Augmentation ist bei TL wichtig gegen Overfitting
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.6, 1.4]
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb'
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        shuffle=False
    )
    return train_gen, val_gen

# ==========================================
# --- TRAINING ROUTINE (2 PHASEN) ---
# ==========================================
def train_transfer_model(params, log_dir, label_suffix):
    """
    Führt den kompletten 2-Phasen Prozess für EINE Konfiguration durch.
    Gibt (History_Combined, Final_Model) zurück.
    """
    # 1. Daten
    train_gen, val_gen = get_generators(params['batch_size'])
    
    # 2. Basis Modell laden
    base_model = get_base_model(CHOSEN_MODEL, INPUT_SHAPE)
    base_model.trainable = False # Erstmal einfrieren
    
    # 3. Kopf bauen
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(params['dense_units'], activation=params['activation']),
        layers.Dropout(params['dropout']),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Optimizer wählen
    if params['optimizer'] == 'adam': opt_cls = optimizers.Adam
    elif params['optimizer'] == 'sgd': opt_cls = optimizers.SGD
    elif params['optimizer'] == 'rmsprop': opt_cls = optimizers.RMSprop
    else: opt_cls = optimizers.Adam

    # --- PHASE 1: WARM-UP ---
    model.compile(optimizer=opt_cls(learning_rate=params['warmup_lr']),
                  loss=params['loss'], metrics=['accuracy'])
    
    csv_path_1 = os.path.join(log_dir, f"log_warmup_{label_suffix}.csv")
    print(f"   [Phase 1] Warm-Up ({params['warmup_epochs']} Epochen)...")
    
    history_1 = model.fit(
        train_gen, epochs=params['warmup_epochs'], validation_data=val_gen, verbose=0,
        callbacks=[CSVLogger(csv_path_1)]
    )
    
    # --- PHASE 2: FINE-TUNING ---
    base_model.trainable = True
    
    # Dynamisches Einfrieren
    layer_count = len(base_model.layers)
    freeze_until = max(0, layer_count - params['unfreeze_layers'])
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
        
    # Neu kompilieren (Low LR)
    model.compile(optimizer=opt_cls(learning_rate=params['finetune_lr']),
                  loss=params['loss'], metrics=['accuracy'])
    
    csv_path_2 = os.path.join(log_dir, f"log_finetune_{label_suffix}.csv")
    callbacks_2 = [
        CSVLogger(csv_path_2),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    ]
    
    print(f"   [Phase 2] Fine-Tuning (Letzte {params['unfreeze_layers']} Layer, LR: {params['finetune_lr']})...")
    
    history_2 = model.fit(
        train_gen, 
        epochs=params['warmup_epochs'] + params['finetune_epochs'], # Total epochs
        initial_epoch=history_1.epoch[-1],
        validation_data=val_gen, 
        callbacks=callbacks_2, 
        verbose=0
    )
    
    # Historien kombinieren für Plotting
    combined_history = {}
    for k in history_1.history.keys():
        combined_history[k] = history_1.history[k] + history_2.history[k]
        
    return combined_history, model

# ==========================================
# --- PLOTTING & REPORTING ---
# ==========================================
def create_comparison_plots(results_list, category_name, save_folder):
    colors = ['black'] + sns.color_palette("bright", len(results_list)-1)

    # 1. HISTORY PLOT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    for idx, res in enumerate(results_list):
        hist = res['history']
        best = max(hist['val_accuracy'])
        label = f"{res['label']} (Best: {best:.2%})"
        style = '--' if idx == 0 else '-'
        width = 3 if idx == 0 else 2
        
        ax1.plot(hist['val_accuracy'], label=label, linestyle=style, linewidth=width, color=colors[idx])
        ax2.plot(hist['val_loss'], label=res['label'], linestyle=style, linewidth=width, color=colors[idx])
        
    # Vertikale Linie für Phase-Wechsel (ungefähr, da Early Stopping variiert)
    warmup_eps = BASE_PARAMS['warmup_epochs']
    ax1.axvline(x=warmup_eps-1, color='gray', linestyle=':', alpha=0.5, label='Start Fine-Tuning')
    ax2.axvline(x=warmup_eps-1, color='gray', linestyle=':', alpha=0.5)

    ax1.set_title(f'Transfer Learning Accuracy: {category_name}')
    ax1.set_ylabel('Val Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title(f'Transfer Learning Loss: {category_name}')
    ax2.set_ylabel('Val Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'History_{category_name}.png'))
    plt.close()

    # DATEN FÜR STATISTIK
    violin_data = []
    bar_data = []
    
    for res in results_list:
        acc_values = res['history']['val_accuracy'][-5:] # Letzte 5 Epochen
        if len(acc_values) == 0: acc_values = [0]
        
        for val in acc_values:
            violin_data.append({'Config': res['label'], 'Accuracy': val})
        
        bar_data.append({'Config': res['label'], 'Best Accuracy': max(res['history']['val_accuracy'])})

    df_violin = pd.DataFrame(violin_data)
    df_bar = pd.DataFrame(bar_data)

    # 2. VIOLIN PLOT
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_violin, x='Config', y='Accuracy', hue='Config', palette=colors, legend=False)
    plt.title(f'Stabilität (Letzte 5 Epochen) - {category_name}')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'Violin_{category_name}.png'))
    plt.close()

    # 3. BAR PLOT
    plt.figure(figsize=(10, 6))
    bp = sns.barplot(data=df_bar, x='Config', y='Best Accuracy', hue='Config', palette=colors, legend=False)
    plt.title(f'Beste Accuracy - {category_name}')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    for p in bp.patches:
        bp.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'Bar_{category_name}.png'))
    plt.close()

def write_evaluation_file(results_list, category_name, save_folder, base_acc):
    path = os.path.join(save_folder, f'Evaluation_{category_name}.txt')
    sorted_res = sorted(results_list, key=lambda x: max(x['history']['val_accuracy']), reverse=True)
    
    with open(path, 'w') as f:
        f.write(f"=== TL AUSWERTUNG: {category_name} ({CHOSEN_MODEL}) ===\n\n")
        f.write(f"Basis-Accuracy: {base_acc:.2%}\n")
        f.write("-" * 50 + "\n")
        
        for idx, res in enumerate(sorted_res):
            best = max(res['history']['val_accuracy'])
            diff = best - base_acc
            marker = " [REFERENZ]" if "Basis" in res['label'] else (f" [{'+' if diff>0 else ''}{diff:.2%}]")
            
            f.write(f"{idx+1}. {res['label']}{marker}\n")
            f.write(f"   Accuracy: {best:.2%}\n")
            
            if "Basis" not in res['label']:
                f.write("   Änderungen:\n")
                for k, v in res['params'].items():
                    if k in BASE_PARAMS and BASE_PARAMS[k] != v:
                        f.write(f"     * {k}: {v} (Basis: {BASE_PARAMS[k]})\n")
            f.write("\n")

# ==========================================
# --- MAIN LOOP ---
# ==========================================
def run_full_experiment():
    if os.path.exists(BASE_MODELS_DIR): shutil.rmtree(BASE_MODELS_DIR)
    if os.path.exists(BASE_LOGS_DIR): shutil.rmtree(BASE_LOGS_DIR)
    os.makedirs(BASE_MODELS_DIR); os.makedirs(BASE_LOGS_DIR)

    print(f"\n=== START TRANSFER LEARNING OPTIMIERUNG ({CHOSEN_MODEL}) ===\n")

    # 1. BASIS MODELL
    print("--- [1/3] BASIS REFERENZ TRAINIEREN ---")
    base_folder = os.path.join(BASE_MODELS_DIR, "Base_Ref")
    os.makedirs(base_folder, exist_ok=True)
    
    hist_base, model_base = train_transfer_model(BASE_PARAMS, base_folder, "base")
    model_base.save(os.path.join(base_folder, "Base_Model.keras"))
    
    base_best_acc = max(hist_base['val_accuracy'])
    base_result = {'label': 'Basis (Ref)', 'history': hist_base, 'params': BASE_PARAMS}
    print(f">> Basis fertig. Accuracy: {base_best_acc:.2%}")

    # 2. EINZEL-TESTS
    print("\n--- [2/3] PARAMETER TESTS ---")
    for param, values in SINGLE_PARAM_TESTS.items():
        print(f"\n>>> Teste: {param}")
        curr_model_dir = os.path.join(BASE_MODELS_DIR, f"{EXPERIMENT_NAME}_{param}")
        curr_log_dir = os.path.join(BASE_LOGS_DIR, f"{EXPERIMENT_NAME}_{param}")
        os.makedirs(curr_model_dir, exist_ok=True); os.makedirs(curr_log_dir, exist_ok=True)
        
        results = [base_result]
        
        for val in values:
            print(f"   -> Wert: {val}")
            new_params = BASE_PARAMS.copy()
            new_params[param] = val
            
            hist, model = train_transfer_model(new_params, curr_log_dir, f"{param}_{val}")
            label = f"{param}={val}"
            model.save(os.path.join(curr_model_dir, f"Model_{label}.keras"))
            
            results.append({'label': label, 'history': hist, 'params': new_params})
            
        create_comparison_plots(results, param, curr_log_dir)
        write_evaluation_file(results, param, curr_log_dir, base_best_acc)

    # 3. KOMBINATIONEN
    print("\n--- [3/3] KOMBINATIONS TESTS ---")
    for group_name, scenarios in COMBINATION_GROUPS.items():
        print(f"\n>>> Gruppe: {group_name}")
        curr_model_dir = os.path.join(BASE_MODELS_DIR, f"{EXPERIMENT_NAME}_{group_name}")
        curr_log_dir = os.path.join(BASE_LOGS_DIR, f"{EXPERIMENT_NAME}_{group_name}")
        os.makedirs(curr_model_dir, exist_ok=True); os.makedirs(curr_log_dir, exist_ok=True)
        
        results = [base_result]
        
        for scen_label, changes in scenarios.items():
            print(f"   -> Szenario: {scen_label}")
            new_params = BASE_PARAMS.copy()
            new_params.update(changes)
            
            hist, model = train_transfer_model(new_params, curr_log_dir, f"{group_name}_{scen_label}")
            model.save(os.path.join(curr_model_dir, f"Model_{scen_label}.keras"))
            
            results.append({'label': scen_label, 'history': hist, 'params': new_params})
            
        create_comparison_plots(results, group_name, curr_log_dir)
        write_evaluation_file(results, group_name, curr_log_dir, base_best_acc)

    print("\n=== FERTIG! ===")

if __name__ == "__main__":
    run_full_experiment()