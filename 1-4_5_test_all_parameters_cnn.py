import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import shutil
import json

# ==========================================
# --- MACOS M4 OPTIMIERUNG ---
# ==========================================
# Accessibility abschalten, um VS Code Abstürze zu verhindern
os.environ['QT_ACCESSIBILITY'] = '0' 

# ==========================================
# --- KONFIGURATION & EINGABE ---
# ==========================================

# Ordner-Struktur
BASE_MODELS_DIR = 'models/Bayesian_1_256,256_allparams_tests'
BASE_LOGS_DIR = 'logs/Bayesian_1_256,256_allparams_tests'
DATASET_DIR = 'dataset_final_boxes'

# Name des Experiments
EXPERIMENT_NAME = "Bayesian_1_256,256_allparams_tests"

# Bild-Parameter
IMG_SIZE = (256, 256)
INPUT_SHAPE = IMG_SIZE + (3,)

# --- 1. DER URSPRUNG (BASIS-WERTE) ---
BASE_PARAMS = {
    'epochs': 25,                       # Festgelegt für den Vergleich
    'batch_size': 16,                   # Tuner
    'learning_rate': 0.0002266,         # Tuner (gerundet)
    'optimizer': 'adam',                # Standard
    'activation': 'relu',               # Standard
    'dropout': 0.0,                     # Tuner
    'dense_units': 320,                 # Tuner
    'conv_blocks': 4,                   # Tuner (num_blocks)
    'start_filters': 32,                # Tuner
    'batch_norm': True,                 # Tuner
    'loss': 'binary_crossentropy'
}

# --- 2. EINZEL-PARAMETER TESTS (Höher / Tiefer) ---
SINGLE_PARAM_TESTS = {
    'batch_size': [8, 32],              # Basis: 16
    'learning_rate': [0.00002, 0.002],  # Basis: 0.0002266
    'dropout': [0.2, 0.5],              # Basis: 0.0
    'dense_units': [128, 512],          # Basis: 320
    'conv_blocks': [2, 5],              # Basis: 4
    'batch_norm': [False],              # Basis: True
    'optimizer': ['sgd', 'rmsprop'],    # Basis: adam
    'activation': ['elu', 'tanh']       # Basis: relu
}

# --- 3. KOMBINATIONS-TESTS (LOGISCHE GRUPPEN) ---
# Hier testen wir "Erhöhung" und "Verringerung" von ganzen Parameter-Paketen
COMBINATION_GROUPS = {
    
   # GRUPPE A: Modell-Komplexität (Netzgröße & Regularisierung)
    'Combo_Capacity': {
        'Increase (+)': { # "Großes Netz" (Mehr Kapazität als Basis)
            'dense_units': 512,      # Basis: 320 -> Erhöht
            'conv_blocks': 5,        # Basis: 4 -> Erhöht (Tieferes Netz)
            'start_filters': 64,     # Basis: 32 -> Erhöht (Breiteres Netz)
            'dropout': 0.4           # Basis: 0.0 -> Erhöht (Nötig gegen Overfitting bei großem Netz)
        },
        'Decrease (-)': { # "Kleines Netz" (Weniger Kapazität als Basis)
            'dense_units': 128,      # Basis: 320 -> Verringert
            'conv_blocks': 3,        # Basis: 4 -> Verringert
            'start_filters': 16,     # Basis: 32 -> Verringert
            'dropout': 0.0           # Basis: 0.0 -> Bleibt 0 (Kleines Netz braucht kaum Regularisierung)
        }
    },

    # GRUPPE B: Trainings-Dynamik (Geschwindigkeit & Schrittweite)
    'Combo_Dynamics': {
        'Increase (+)': { # "Aggressives Lernen" (Schneller & Gröber)
            'batch_size': 32,        # Basis: 16 -> Erhöht (stabilerer Gradient, schneller)
            'learning_rate': 0.002   # Basis: ~0.0002 -> Erhöht (Faktor 10)
        },
        'Decrease (-)': { # "Konservatives Lernen" (Langsamer & Feiner)
            'batch_size': 8,         # Basis: 16 -> Verringert (mehr Updates pro Epoche)
            'learning_rate': 0.00002 # Basis: ~0.0002 -> Verringert (Faktor 10 kleiner)
        }
    }
}

# ==========================================
# --- DATA GENERATOR ---
# ==========================================
def get_generators(batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # shear_range=0.1, 
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
# --- MODELL BAUEN ---
# ==========================================
def create_model(params):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=INPUT_SHAPE))
    
    current_filters = params['start_filters']
    
    # Convolutional Blocks
    for i in range(params['conv_blocks']):
        model.add(layers.Conv2D(current_filters, (3, 3), padding='same'))
        model.add(layers.Activation(params['activation']))
        model.add(layers.MaxPooling2D((2, 2)))
        current_filters *= 2
    
    model.add(layers.Flatten())
    
    # Dense Part
    model.add(layers.Dense(params['dense_units']))
    model.add(layers.Activation(params['activation']))
    
    if params['dropout'] > 0:
        model.add(layers.Dropout(params['dropout']))
        
    model.add(layers.Dense(1, activation='sigmoid'))

    # Optimizer
    if params['optimizer'] == 'adam': opt = optimizers.Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'sgd': opt = optimizers.SGD(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop': opt = optimizers.RMSprop(learning_rate=params['learning_rate'])
    else: opt = optimizers.Adam(learning_rate=params['learning_rate'])

    model.compile(optimizer=opt, loss=params['loss'], metrics=['accuracy'])
    return model

# ==========================================
# --- PLOTTING & LOGGING ---
# ==========================================
def create_comparison_plots(results_list, category_name, save_folder):
    """
    Erstellt alle Plots.
    results_list[0] ist IMMER das Basis-Modell.
    """
    # Farben: Basis = Schwarz, Varianten = Grün/Rot Töne oder Bunt
    colors = ['black'] + sns.color_palette("bright", len(results_list)-1)

    # 1. HISTORY PLOT (Verlauf)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Accuracy Verlauf
    for idx, res in enumerate(results_list):
        best = max(res['history']['val_accuracy'])
        label = f"{res['label']} (Best: {best:.2%})"
        style = '--' if idx == 0 else '-' # Basis gestrichelt
        width = 3 if idx == 0 else 2.5
        ax1.plot(res['history']['val_accuracy'], label=label, linestyle=style, linewidth=width, color=colors[idx])
        
    ax1.set_title(f'Verlauf Accuracy: {category_name}')
    ax1.set_ylabel('Val Accuracy')
    ax1.set_xlabel('Epochen')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss Verlauf
    for idx, res in enumerate(results_list):
        style = '--' if idx == 0 else '-'
        width = 3 if idx == 0 else 2.5
        ax2.plot(res['history']['val_loss'], label=res['label'], linestyle=style, linewidth=width, color=colors[idx])
        
    ax2.set_title(f'Verlauf Loss: {category_name}')
    ax2.set_ylabel('Val Loss')
    ax2.set_xlabel('Epochen')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'History_Comparison_{category_name}.png'))
    plt.close()

    # DATEN VORBEREITEN
    violin_data = []
    bar_data = []
    
    for res in results_list:
        # Violin: Letzte 5 Epochen für Stabilität
        acc_values = res['history']['val_accuracy'][-5:] 
        if len(acc_values) < 5: acc_values = res['history']['val_accuracy'] 
        
        for val in acc_values:
            violin_data.append({'Config': res['label'], 'Accuracy': val})
            
        bar_data.append({'Config': res['label'], 'Best Accuracy': max(res['history']['val_accuracy'])})

    df_violin = pd.DataFrame(violin_data)
    df_bar = pd.DataFrame(bar_data)

    # 2. VIOLIN PLOT (Verteilung + Vergleich mit Basis)
    plt.figure(figsize=(10, 6))
    # Basis ist immer links, da results_list[0] Basis ist
    sns.violinplot(data=df_violin, x='Config', y='Accuracy', hue='Config', palette=colors, legend=False)
    plt.title(f'Verteilung der Accuracy (Letzte 5 Epochen) - {category_name}')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'Violin_Plot_{category_name}.png'))
    plt.close()

    # 3. BAR PLOT (Ranking)
    plt.figure(figsize=(10, 6))
    bp = sns.barplot(data=df_bar, x='Config', y='Best Accuracy', hue='Config', palette=colors, legend=False)
    plt.title(f'Maximale erreichte Accuracy - {category_name}')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    
    for p in bp.patches:
        bp.annotate(f"{p.get_height():.2%}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
                   
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'Bar_Plot_{category_name}.png'))
    plt.close()

def write_evaluation_file(results_list, category_name, save_folder, base_acc):
    """Schreibt die Textauswertung mit Diff zum Original."""
    path = os.path.join(save_folder, f'Evaluation_{category_name}.txt')
    
    # Sortieren nach Performance
    sorted_res = sorted(results_list, key=lambda x: max(x['history']['val_accuracy']), reverse=True)
    
    with open(path, 'w') as f:
        f.write(f"=== AUSWERTUNG: {category_name} ===\n\n")
        f.write(f"Basis-Modell Accuracy: {base_acc:.2%}\n")
        f.write("-" * 50 + "\n")
        
        for idx, res in enumerate(sorted_res):
            best = max(res['history']['val_accuracy'])
            diff = best - base_acc
            loss = min(res['history']['val_loss'])
            
            marker = ""
            if "Basis" in res['label']: marker = " [REFERENZ]"
            elif diff > 0.005: marker = " [VERBESSERUNG]"
            elif diff < -0.005: marker = " [VERSCHLECHTERUNG]"
            
            f.write(f"{idx+1}. {res['label']}{marker}\n")
            f.write(f"   - Beste Accuracy: {best:.2%}\n")
            f.write(f"   - Differenz zur Basis: {diff:+.2%}\n")
            f.write(f"   - Bester Loss: {loss:.4f}\n")
            
            if "Basis" not in res['label']:
                f.write("   - Geänderte Parameter:\n")
                for k, v in res['params'].items():
                    if k in BASE_PARAMS and BASE_PARAMS[k] != v:
                        f.write(f"     * {k}: {v} (Basis: {BASE_PARAMS[k]})\n")
            f.write("\n")

# ==========================================
# --- MAIN LOOP ---
# ==========================================
def run_full_experiment():
    # Aufräumen
    if os.path.exists(BASE_MODELS_DIR): shutil.rmtree(BASE_MODELS_DIR)
    if os.path.exists(BASE_LOGS_DIR): shutil.rmtree(BASE_LOGS_DIR)
    
    os.makedirs(BASE_MODELS_DIR)
    os.makedirs(BASE_LOGS_DIR)

    print("\n" + "="*50)
    print(f"   START: {EXPERIMENT_NAME} (Auf M4 Silicon)")
    print("="*50)

    # 1. BASIS MODELL TRAINIEREN (Referenz für ALLE Tests)
    print("\n--- [1/3] BASIS-MODELL TRAINIEREN ---")
    
    base_gen_train, base_gen_val = get_generators(BASE_PARAMS['batch_size'])
    base_model = create_model(BASE_PARAMS)
    
    history_base = base_model.fit(
        base_gen_train, 
        epochs=BASE_PARAMS['epochs'], 
        validation_data=base_gen_val, 
        verbose=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    
    base_best_acc = max(history_base.history['val_accuracy'])
    print(f">> Basis fertig. Accuracy: {base_best_acc:.2%}")
    
    # Basis-Ergebnis speichern
    base_result = {
        'label': 'Basis (Original)',
        'history': history_base.history,
        'params': BASE_PARAMS
    }

    # 2. EINZEL-PARAMETER TESTS
    print("\n--- [2/3] EINZEL-PARAMETER TESTS ---")
    for param, values in SINGLE_PARAM_TESTS.items():
        print(f"\n>>> Teste: {param} <<<")
        
        current_model_dir = os.path.join(BASE_MODELS_DIR, f"{EXPERIMENT_NAME}_{param}")
        current_log_dir = os.path.join(BASE_LOGS_DIR, f"{EXPERIMENT_NAME}_{param}")
        os.makedirs(current_model_dir, exist_ok=True)
        os.makedirs(current_log_dir, exist_ok=True)
        
        results = [base_result] # Basis ist immer der erste Eintrag
        
        for val in values:
            print(f"   -> Wert: {val}")
            
            new_params = BASE_PARAMS.copy()
            new_params[param] = val
            
            t_gen, v_gen = get_generators(new_params['batch_size'])
            model = create_model(new_params)
            
            csv_path = os.path.join(current_log_dir, f"log_{param}_{val}.csv")
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                CSVLogger(csv_path)
            ]
            
            hist = model.fit(
                t_gen, epochs=new_params['epochs'], validation_data=v_gen, 
                callbacks=callbacks, verbose=1
            )
            
            label = f"{param}={val}"
            model.save(os.path.join(current_model_dir, f"Model_{label}.keras"))
            
            results.append({
                'label': label,
                'history': hist.history,
                'params': new_params
            })
            
        create_comparison_plots(results, param, current_log_dir)
        write_evaluation_file(results, param, current_log_dir, base_best_acc)

    # 3. KOMBINATIONS-TESTS (LOGISCHE GRUPPEN)
    print("\n--- [3/3] KOMBINATIONS-TESTS (High vs. Low) ---")
    
    for group_name, scenarios in COMBINATION_GROUPS.items():
        print(f"\n>>> Teste Gruppe: {group_name} <<<")
        
        current_model_dir = os.path.join(BASE_MODELS_DIR, f"{EXPERIMENT_NAME}_{group_name}")
        current_log_dir = os.path.join(BASE_LOGS_DIR, f"{EXPERIMENT_NAME}_{group_name}")
        os.makedirs(current_model_dir, exist_ok=True)
        os.makedirs(current_log_dir, exist_ok=True)
        
        results = [base_result] # Auch hier gegen Basis vergleichen
        
        # Iteriere durch "Increase (+)" und "Decrease (-)"
        for scenario_label, param_changes in scenarios.items():
            print(f"   -> Szenario: {scenario_label}")
            
            new_params = BASE_PARAMS.copy()
            new_params.update(param_changes)
            
            t_gen, v_gen = get_generators(new_params['batch_size'])
            model = create_model(new_params)
            
            csv_path = os.path.join(current_log_dir, f"log_{group_name}_{scenario_label}.csv")
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                CSVLogger(csv_path)
            ]
            
            hist = model.fit(
                t_gen, epochs=new_params['epochs'], validation_data=v_gen, 
                callbacks=callbacks, verbose=1
            )
            
            model.save(os.path.join(current_model_dir, f"Model_{group_name}_{scenario_label}.keras"))
            
            results.append({
                'label': scenario_label, # Heißt dann z.B. "Increase (+)"
                'history': hist.history,
                'params': new_params
            })
            
        create_comparison_plots(results, group_name, current_log_dir)
        write_evaluation_file(results, group_name, current_log_dir, base_best_acc)

    print("\n" + "="*50)
    print("   ALLE TESTS ERFOLGREICH ABGESCHLOSSEN.")
    print("="*50)

if __name__ == "__main__":
    run_full_experiment()