import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16 # type: ignore
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
# --- KONFIGURATION ---
# ==========================================
# HIER WÄHLEN: Welches Modell soll getestet werden?
CHOSEN_MODEL = "MobileNetV2" 
# CHOSEN_MODEL = "ResNet50"
# CHOSEN_MODEL = "VGG16"

BASE_MODELS_DIR = f'models/TFL_Tests_{CHOSEN_MODEL}'
BASE_LOGS_DIR = f'logs/TFL_Tests_{CHOSEN_MODEL}'
DATASET_DIR = 'dataset_final_boxes'
IMG_SIZE = (224, 224)
INPUT_SHAPE = IMG_SIZE + (3,)

# --- BASIS HYPERPARAMETER (Deine Gewinner-Werte) ---
CONFIGS = {
    "MobileNetV2": {
        'warmup_epochs': 10, 'warmup_lr': 0.001, 'warmup_opt': 'rmsprop',
        'finetune_epochs': 25, 'finetune_lr': 5e-05, 'unfreeze_layers': 20,
        'batch_size': 32, 'dropout': 0.0, 'dense_units': 320
    },
    "ResNet50": {
        'warmup_epochs': 10, 'warmup_lr': 0.001, 'warmup_opt': 'adam',
        'finetune_epochs': 25, 'finetune_lr': 1e-05, 'unfreeze_layers': 150,
        'batch_size': 32, 'dropout': 0.3, 'dense_units': 256
    },
    "VGG16": {
        'warmup_epochs': 10, 'warmup_lr': 0.0001, 'warmup_opt': 'adam',
        'finetune_epochs': 25, 'finetune_lr': 1e-05, 'unfreeze_layers': 15,
        'batch_size': 32, 'dropout': 0.5, 'dense_units': 512
    }
}
BASE_PARAMS = CONFIGS[CHOSEN_MODEL]

# --- PARAMETER TESTS (Abweichungen von der Basis) ---
SINGLE_PARAM_TESTS = {
    'finetune_lr': [1e-6, 1e-4],      
    'unfreeze_layers': [10, 40] if CHOSEN_MODEL == "MobileNetV2" else [50, 100],      
    'dropout': [0.2, 0.5] if BASE_PARAMS['dropout'] == 0.0 else [0.0, 0.6],            
    'dense_units': [128, 512],        
    'batch_size': [16, 64]
}

COMBINATION_GROUPS = {
    'Combo_Capacity': { 
        'Increase (+)': { 'dense_units': 512, 'dropout': 0.5, 'unfreeze_layers': BASE_PARAMS['unfreeze_layers'] + 20 },
        'Decrease (-)': { 'dense_units': 128, 'dropout': 0.0, 'unfreeze_layers': max(5, BASE_PARAMS['unfreeze_layers'] - 10) }
    }
}

# ==========================================
# --- HELFER ---
# ==========================================
def get_base_model(name, input_shape):
    if name == "MobileNetV2": return MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == "ResNet50": return ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == "VGG16": return VGG16(input_shape=input_shape, include_top=False, weights='imagenet')

def get_generators(batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=15, width_shift_range=0.1, 
        height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'train'), target_size=IMG_SIZE, batch_size=batch_size, 
        class_mode='binary', shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'val'), target_size=IMG_SIZE, batch_size=batch_size, 
        class_mode='binary', shuffle=False
    )
    return train_gen, val_gen

def get_optimizer(name, lr):
    if name == 'adam': return optimizers.Adam(learning_rate=lr)
    elif name == 'sgd': return optimizers.SGD(learning_rate=lr)
    elif name == 'rmsprop': return optimizers.RMSprop(learning_rate=lr)
    return optimizers.Adam(learning_rate=lr)

# ==========================================
# --- TRAINING ---
# ==========================================
def train_model(params, log_dir, label):
    train_gen, val_gen = get_generators(params['batch_size'])
    base_model = get_base_model(CHOSEN_MODEL, INPUT_SHAPE)
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(params['dense_units'], activation='relu'),
        layers.Dropout(params['dropout']),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Phase 1: Warmup
    model.compile(optimizer=get_optimizer(params.get('warmup_opt', 'adam'), params['warmup_lr']), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    hist1 = model.fit(train_gen, epochs=params['warmup_epochs'], validation_data=val_gen, verbose=0)
    
    # Phase 2: Fine-Tuning
    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - params['unfreeze_layers'])
    for layer in base_model.layers[:freeze_until]: layer.trainable = False
    
    model.compile(optimizer=optimizers.Adam(learning_rate=params['finetune_lr']), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0),
        CSVLogger(os.path.join(log_dir, f"log_{label}.csv"))
    ]
    
    hist2 = model.fit(
        train_gen, epochs=params['warmup_epochs'] + params['finetune_epochs'], 
        initial_epoch=hist1.epoch[-1] + 1, validation_data=val_gen, callbacks=callbacks, verbose=0
    )
    
    combined_history = {}
    for k in hist1.history.keys():
        combined_history[k] = hist1.history[k] + hist2.history[k]
        
    return combined_history, model, val_gen

# ==========================================
# --- PLOTTING ---
# ==========================================
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
BASE_COLOR = '#000000'
PLOT_PALETTE = sns.color_palette("bright", n_colors=12)

def create_comparison_plots(results_list, category_name, save_folder):
    plot_colors = [BASE_COLOR] + PLOT_PALETTE[:len(results_list)-1]
    
    # 1. History Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    for idx, res in enumerate(results_list):
        best = max(res['history']['val_accuracy'])
        label = f"{res['label']} (Best: {best:.2%})"
        style = '--' if idx == 0 else '-'
        width = 2.5 if idx == 0 else 2
        ax1.plot(res['history']['val_accuracy'], label=label, linestyle=style, linewidth=width, color=plot_colors[idx], alpha=0.9)
    
    ax1.axvline(x=BASE_PARAMS['warmup_epochs']-1, color='gray', linestyle=':', linewidth=2, label='Fine-Tuning')
    ax1.set_title(f'Verlauf Accuracy: {category_name}', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    for idx, res in enumerate(results_list):
        style = '--' if idx == 0 else '-'
        width = 2.5 if idx == 0 else 2
        ax2.plot(res['history']['val_loss'], label=res['label'], linestyle=style, linewidth=width, color=plot_colors[idx], alpha=0.9)
    
    ax2.axvline(x=BASE_PARAMS['warmup_epochs']-1, color='gray', linestyle=':', linewidth=2, label='Fine-Tuning')
    ax2.set_title(f'Verlauf Loss: {category_name}', fontweight='bold')
    ax2.set_ylabel('Loss')
    
    plt.tight_layout(); plt.savefig(os.path.join(save_folder, f'History_Comparison_{category_name}.png'), dpi=300); plt.close()

    # Daten vorbereiten
    violin_df_list = []
    bar_data = []
    for res in results_list:
        bar_data.append({'Config': res['label'], 'Best Accuracy': max(res['history']['val_accuracy'])})
        probs = res['y_pred_prob'].flatten()
        for p, t in zip(probs, res['y_true']):
            violin_df_list.append({'Config': res['label'], 'Probability': p, 'True Class': 'QR (1)' if t == 1 else 'No QR (0)'})

    # 2. Violin Plot
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=pd.DataFrame(violin_df_list), x='Config', y='Probability', hue='True Class', 
                   split=True, inner='quart', palette={"No QR (0)": "skyblue", "QR (1)": "orange"}, linewidth=1.2, cut=0)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold 0.5')
    plt.title(f'Vorhersage-Sicherheit (Split by Class) - {category_name}', fontweight='bold')
    plt.ylim(-0.05, 1.05)
    plt.xticks(rotation=15, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(os.path.join(save_folder, f'Violin_Split_Plot_{category_name}.png'), dpi=300); plt.close()

    # 3. Bar Plot
    plt.figure(figsize=(10, len(results_list) * 0.8 + 2))
    bp = sns.barplot(data=pd.DataFrame(bar_data), y='Config', x='Best Accuracy', palette=plot_colors, hue='Config', legend=False, orient='h')
    plt.title(f'Maximale Accuracy - {category_name}', fontweight='bold')
    plt.xlim(0, 1.05); plt.grid(True, axis='x', alpha=0.7)
    for p in bp.patches:
        plt.text(p.get_width()+0.01, p.get_y()+p.get_height()/2, f'{p.get_width():.2%}', va='center', fontweight='bold', color='black')
    plt.tight_layout(); plt.savefig(os.path.join(save_folder, f'Bar_Plot_Horizontal_{category_name}.png'), dpi=300); plt.close()

def create_confusion_matrices(results_list, category_name, save_folder):
    """Erstellt Konfusionsmatrizen Vergleich (Basis vs. Variante)."""
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
        ax1.set_title(f"Basis (Winner)\nAcc: {max(base_res['history']['val_accuracy']):.2%}", fontweight='bold')
        ax1.set_ylabel('Wahrheit'); ax1.set_xlabel('Vorhersage')

        sns.heatmap(comp_cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=labels, yticklabels=labels, ax=ax2, annot_kws={"size": 14})
        ax2.set_title(f"Variante: {comp_res['label']}\nAcc: {max(comp_res['history']['val_accuracy']):.2%}", fontweight='bold')
        ax2.set_ylabel('Wahrheit'); ax2.set_xlabel('Vorhersage')
        
        plt.suptitle(f'Confusion Matrix Vergleich: {category_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        safe_label = "".join([c if c.isalnum() else "_" for c in comp_res['label']])
        plt.savefig(os.path.join(save_folder, f'CM_Compare_{safe_label}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def write_evaluation(results_list, category_name, save_folder, base_acc):
    with open(os.path.join(save_folder, f'Evaluation_{category_name}.txt'), 'w') as f:
        f.write(f"=== AUSWERTUNG {CHOSEN_MODEL}: {category_name} ===\n\n")
        f.write(f"Basis-Modell Accuracy: {base_acc:.2%}\n")
        f.write("-" * 50 + "\n")
        
        for idx, res in sorted(enumerate(results_list), key=lambda x: max(x[1]['history']['val_accuracy']), reverse=True):
            res_data = res[1] 
            best = max(res_data['history']['val_accuracy'])
            diff = best - base_acc
            marker = " [REFERENZ]" if "Basis" in res_data['label'] else ""
            
            f.write(f"{res_data['label']}{marker}\n")
            f.write(f"   - Beste Accuracy: {best:.2%}\n")
            f.write(f"   - Differenz: {diff:+.2%}\n\n")

# ==========================================
# --- MAIN LOOP ---
# ==========================================
def run_full_experiment():
    if os.path.exists(BASE_MODELS_DIR): shutil.rmtree(BASE_MODELS_DIR)
    if os.path.exists(BASE_LOGS_DIR): shutil.rmtree(BASE_LOGS_DIR)
    os.makedirs(BASE_MODELS_DIR); os.makedirs(BASE_LOGS_DIR)

    print(f"\nSTART TFL TESTS: {CHOSEN_MODEL}\n{'='*40}")

    # 1. BASIS
    print("\n--- BASIS MODELL (REFERENZ) ---")
    hist_base, model_base, val_gen_base = train_model(BASE_PARAMS, BASE_LOGS_DIR, "Base")
    
    # HIER IST DIE ÄNDERUNG: Basis-Modell speichern
    model_base.save(os.path.join(BASE_MODELS_DIR, f"Base_Winner_{CHOSEN_MODEL}.keras"))
    
    val_gen_base.reset()
    base_y_pred = model_base.predict(val_gen_base, verbose=0)
    base_best_acc = max(hist_base['val_accuracy'])
    
    base_result = {
        'label': 'Basis (Winner)', 'history': hist_base, 'params': BASE_PARAMS,
        'y_pred_prob': base_y_pred, 'y_true': val_gen_base.classes
    }
    print(f">> Basis Accuracy: {base_best_acc:.2%}")

    # 2. TESTS
    all_tests = {}
    all_tests.update(SINGLE_PARAM_TESTS)
    all_tests.update(COMBINATION_GROUPS)

    for cat_name, items in all_tests.items():
        print(f"\n>>> Kategorie: {cat_name}")
        curr_log_dir = os.path.join(BASE_LOGS_DIR, cat_name)
        curr_model_dir = os.path.join(BASE_MODELS_DIR, cat_name) # Ordner für Modelle
        os.makedirs(curr_log_dir, exist_ok=True)
        os.makedirs(curr_model_dir, exist_ok=True)
        
        results = [base_result]
        iterable = items.items() if isinstance(items, dict) else [(val, val) for val in items]

        for label_suffix, val_or_dict in iterable:
            if isinstance(items, dict): label = label_suffix; changes = val_or_dict
            else: label = f"{cat_name}={val_or_dict}"; changes = {cat_name: val_or_dict}
            
            print(f"   -> Test: {label}")
            new_params = BASE_PARAMS.copy()
            new_params.update(changes)
            
            hist, model, val_gen = train_model(new_params, curr_log_dir, label.replace('=', '_'))
            
            # HIER IST DIE ÄNDERUNG: Test-Modell speichern
            model.save(os.path.join(curr_model_dir, f"Model_{label.replace('=', '_')}.keras"))
            
            val_gen.reset()
            y_pred = model.predict(val_gen, verbose=0)
            
            results.append({
                'label': label, 'history': hist, 'params': new_params,
                'y_pred_prob': y_pred, 'y_true': val_gen.classes
            })
            
        create_comparison_plots(results, cat_name, curr_log_dir)
        create_confusion_matrices(results, cat_name, curr_log_dir) 
        write_evaluation(results, cat_name, curr_log_dir, base_best_acc)

    print("\nALLE TFL TESTS ABGESCHLOSSEN.")

if __name__ == "__main__":
    run_full_experiment()