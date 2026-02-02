import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16, EfficientNetB0, InceptionV3 # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import shutil
import random

# ==========================================
# --- REPRODUZIERBARKEIT ---
# ==========================================
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

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

# --- MODELL AUSWAHL ---
# CHOSEN_MODEL = "VGG16"
# CHOSEN_MODEL = "ResNet50"
CHOSEN_MODEL = "MobileNetV2" 

def get_base_model(name, input_shape):
    """Lädt die gewählte Architektur ohne Top-Layer."""
    if name == "MobileNetV2": return MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == "ResNet50": return ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == "VGG16": return VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    else: raise ValueError(f"Modell {name} nicht definiert.")

# --- 1. BASIS HYPERPARAMETER ---
BASE_PARAMS = {
    # Phase 1: Warm-Up
    'warmup_epochs': 10,
    'warmup_lr': 0.001,       
    
    # Phase 2: Fine-Tuning
    'finetune_epochs': 15,
    'finetune_lr': 1e-5,      
    'unfreeze_layers': 80,    
    
    # Architektur
    'batch_size': 32,
    'optimizer': 'adam',      
    'activation': 'relu',
    'dropout': 0.2,           
    'dense_units': 384,       
    'loss': 'binary_crossentropy'
}

# --- 2. EINZEL-PARAMETER TESTS ---
SINGLE_PARAM_TESTS = {
    'finetune_lr': [1e-6, 1e-4],      
    'unfreeze_layers': [10, 50],      
    'dropout': [0.3, 0.7],            
    'dense_units': [128, 512],        
    'batch_size': [16, 64],           
    'optimizer': ['rmsprop', 'sgd']   
}

# --- 3. KOMBINATIONS-TESTS ---
COMBINATION_GROUPS = {
    'Combo_Deep_Dive': { 
        'Increase (+)': { 
            'unfreeze_layers': 60, 'finetune_lr': 5e-5, 'dropout': 0.6
        },
        'Decrease (-)': { 
            'unfreeze_layers': 10, 'finetune_lr': 1e-6, 'dropout': 0.3
        }
    },
    'Combo_Capacity': { 
        'Increase (+)': { 
            'dense_units': 512, 'dropout': 0.6
        },
        'Decrease (-)': { 
            'dense_units': 64, 'dropout': 0.2
        }
    }
}

# ==========================================
# --- DATA GENERATOR ---
# ==========================================
def get_generators(batch_size):
    # Die festgelegte Augmentation (ohne Brightness/Shear)
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

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb'
    )
    # WICHTIG: shuffle=False für korrekte Auswertung (CM, Violin)
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
    Führt den kompletten 2-Phasen Prozess durch.
    Gibt (History_Combined, Final_Model, Val_Gen) zurück.
    """
    # 1. Daten
    train_gen, val_gen = get_generators(params['batch_size'])
    
    # 2. Basis Modell laden
    base_model = get_base_model(CHOSEN_MODEL, INPUT_SHAPE)
    base_model.trainable = False 
    
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
        epochs=params['warmup_epochs'] + params['finetune_epochs'], 
        initial_epoch=history_1.epoch[-1],
        validation_data=val_gen, 
        callbacks=callbacks_2, 
        verbose=0
    )
    
    # Historien kombinieren
    combined_history = {}
    for k in history_1.history.keys():
        combined_history[k] = history_1.history[k] + history_2.history[k]
        
    return combined_history, model, val_gen

# ==========================================
# --- NEUE PLOTTING FUNKTIONEN (High-End) ---
# ==========================================
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
BASE_COLOR = '#000000'
# Neue Farbpalette für bessere Unterscheidbarkeit
PLOT_PALETTE = sns.color_palette("bright", n_colors=10)

def create_comparison_plots(results_list, category_name, save_folder):
    """Erstellt History, Split-Violin und horizontale Bar Plots."""
    plot_colors = [BASE_COLOR] + PLOT_PALETTE[:len(results_list)-1]

    # --- 1. HISTORY PLOT ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Vertikale Linie für Phase-Wechsel
    warmup_eps = BASE_PARAMS['warmup_epochs']

    for idx, res in enumerate(results_list):
        best = max(res['history']['val_accuracy'])
        label = f"{res['label']} (Best: {best:.2%})"
        style = '--' if idx == 0 else '-'
        width = 2.5 if idx == 0 else 2
        ax1.plot(res['history']['val_accuracy'], label=label, linestyle=style, linewidth=width, color=plot_colors[idx])
        
    ax1.axvline(x=warmup_eps-1, color='gray', linestyle=':', alpha=0.5, label='Start Fine-Tuning')
    ax1.set_title(f'Verlauf Accuracy: {category_name}', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epochen')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    for idx, res in enumerate(results_list):
        style = '--' if idx == 0 else '-'
        width = 2.5 if idx == 0 else 2
        ax2.plot(res['history']['val_loss'], label=res['label'], linestyle=style, linewidth=width, color=plot_colors[idx])
    
    ax2.axvline(x=warmup_eps-1, color='gray', linestyle=':', alpha=0.5)
    ax2.set_title(f'Verlauf Loss: {category_name}', fontweight='bold')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochen')
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'History_{category_name}.png'), dpi=300)
    plt.close()

    # --- DATEN VORBEREITEN ---
    violin_df_list = []
    bar_data = []
    
    for res in results_list:
        bar_data.append({'Config': res['label'], 'Best Accuracy': max(res['history']['val_accuracy'])})
        
        # Für Split-Violin
        probs = res['y_pred_prob'].flatten()
        true_labels = res['y_true']
        for p, t in zip(probs, true_labels):
            violin_df_list.append({
                'Config': res['label'],
                'Probability': p,
                'True Class': 'QR (1)' if t == 1 else 'No QR (0)'
            })

    df_bar = pd.DataFrame(bar_data)
    df_violin = pd.DataFrame(violin_df_list)

    # --- 2. SPLIT-VIOLIN PLOT ---
    plt.figure(figsize=(14, 7))
    sns.violinplot(data=df_violin, x='Config', y='Probability', hue='True Class', 
                   split=True, inner='quart', palette={"No QR (0)": "skyblue", "QR (1)": "orange"},
                   linewidth=1.2)
    
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    
    plt.title(f'Verteilung der Vorhersage-Sicherheit (Split by Class) - {category_name}', fontweight='bold')
    plt.ylabel('Vorhersagewahrscheinlichkeit')
    plt.xlabel('Konfiguration')
    plt.ylim(-0.05, 1.05)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Wahrheit', loc='upper right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'Violin_Split_{category_name}.png'), dpi=300)
    plt.close()

    # --- 3. HORIZONTAL BAR PLOT ---
    plt.figure(figsize=(10, len(results_list) * 0.8 + 2))
    bp = sns.barplot(data=df_bar, y='Config', x='Best Accuracy', hue='Config', palette=plot_colors, legend=False, orient='h')
    
    plt.title(f'Maximale erreichte Accuracy - {category_name}', fontweight='bold')
    plt.xlabel('Best Accuracy')
    plt.xlim(0, 1.05)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    for p in bp.patches:
        width = p.get_width()
        plt.text(width + 0.01, p.get_y() + p.get_height() / 2.,
                 f'{width:.2%}', 
                 ha='left', va='center', fontweight='bold', color='black')
                 
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'Bar_Horizontal_{category_name}.png'), dpi=300)
    plt.close()

def create_confusion_matrices(results_list, category_name, save_folder):
    """Erstellt Konfusionsmatrizen im Vergleich zur Basis."""
    base_res = results_list[0]
    base_y_pred = (base_res['y_pred_prob'] > 0.5).astype(int).flatten()
    base_cm = confusion_matrix(base_res['y_true'], base_y_pred)

    for i in range(1, len(results_list)):
        comp_res = results_list[i]
        comp_y_pred = (comp_res['y_pred_prob'] > 0.5).astype(int).flatten()
        comp_cm = confusion_matrix(comp_res['y_true'], comp_y_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        labels = ['No QR', 'QR']
        
        sns.heatmap(base_cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=labels, yticklabels=labels, ax=ax1, annot_kws={"size": 14})
        ax1.set_title(f"Basis (Original)\nAcc: {max(base_res['history']['val_accuracy']):.2%}", fontweight='bold')
        ax1.set_ylabel('Wahrheit')
        ax1.set_xlabel('Vorhersage')

        sns.heatmap(comp_cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=labels, yticklabels=labels, ax=ax2, annot_kws={"size": 14})
        ax2.set_title(f"Variante: {comp_res['label']}\nAcc: {max(comp_res['history']['val_accuracy']):.2%}", fontweight='bold')
        ax2.set_ylabel('Wahrheit')
        ax2.set_xlabel('Vorhersage')
        
        plt.suptitle(f'Confusion Matrix Vergleich: {category_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        safe_label = "".join([c if c.isalnum() else "_" for c in comp_res['label']])
        plt.savefig(os.path.join(save_folder, f'CM_Compare_{safe_label}.png'), dpi=300, bbox_inches='tight')
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
    
    hist_base, model_base, val_gen_base = train_transfer_model(BASE_PARAMS, base_folder, "base")
    model_base.save(os.path.join(base_folder, "Base_Model.keras"))
    
    # Vorhersagen Basis
    print("   Generiere Vorhersagen (Basis)...")
    val_gen_base.reset()
    base_y_pred_prob = model_base.predict(val_gen_base, verbose=0)
    base_y_true = val_gen_base.classes

    base_best_acc = max(hist_base['val_accuracy'])
    base_result = {
        'label': 'Basis (Ref)', 
        'history': hist_base, 
        'params': BASE_PARAMS,
        'y_pred_prob': base_y_pred_prob,
        'y_true': base_y_true
    }
    print(f">> Basis fertig. Accuracy: {base_best_acc:.2%}")

    # 2. & 3. TESTS
    all_tests = {}
    all_tests.update(SINGLE_PARAM_TESTS)
    all_tests.update(COMBINATION_GROUPS)

    print("\n--- PARAMETER & KOMBINATIONS TESTS ---")
    for category_name, items in all_tests.items():
        print(f"\n>>> Teste Kategorie: {category_name} <<<")
        curr_model_dir = os.path.join(BASE_MODELS_DIR, f"{EXPERIMENT_NAME}_{category_name}")
        curr_log_dir = os.path.join(BASE_LOGS_DIR, f"{EXPERIMENT_NAME}_{category_name}")
        os.makedirs(curr_model_dir, exist_ok=True); os.makedirs(curr_log_dir, exist_ok=True)
        
        results = [base_result]
        
        # Unterscheidung ob Einzeltest (Liste) oder Kombi (Dict)
        iterable = items.items() if isinstance(items, dict) else [(val, val) for val in items]

        for label_suffix, val_or_dict in iterable:
            if isinstance(items, dict): # Kombi-Test
                label = label_suffix
                param_changes = val_or_dict
            else: # Einzeltest
                label = f"{category_name}={val_or_dict}"
                param_changes = {category_name: val_or_dict}

            print(f"   -> Szenario: {label}")
            new_params = BASE_PARAMS.copy()
            new_params.update(param_changes)
            
            hist, model, val_gen = train_transfer_model(new_params, curr_log_dir, f"{category_name}_{label_suffix}".replace('=', '_'))
            model.save(os.path.join(curr_model_dir, f"Model_{label.replace('=', '_')}.keras"))
            
            # Vorhersagen
            print("      Generiere Vorhersagen...")
            val_gen.reset()
            y_pred_prob = model.predict(val_gen, verbose=0)
            y_true = val_gen.classes

            results.append({
                'label': label, 
                'history': hist, 
                'params': new_params,
                'y_pred_prob': y_pred_prob,
                'y_true': y_true
            })
            
        create_comparison_plots(results, category_name, curr_log_dir)
        create_confusion_matrices(results, category_name, curr_log_dir)
        write_evaluation_file(results, category_name, curr_log_dir, base_best_acc)

    print("\n=== FERTIG! ===")

if __name__ == "__main__":
    run_full_experiment()