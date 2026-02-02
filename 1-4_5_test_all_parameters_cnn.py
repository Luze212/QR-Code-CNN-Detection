import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping # type: ignore
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
# --- KONFIGURATION & EINGABE ---
# ==========================================

# Ordner-Struktur
BASE_MODELS_DIR = 'models/Bayesian_1_256,256_allparams_tests'
BASE_LOGS_DIR = 'logs/Bayesian_1_256,256_allparams_tests'
DATASET_DIR = 'dataset_final_boxes'

EXPERIMENT_NAME = "Bayesian_1_256,256_allparams_tests"

IMG_SIZE = (256, 256)
INPUT_SHAPE = IMG_SIZE + (3,)

# --- 1. BASIS-WERTE (CNN) ---
BASE_PARAMS = {
    'epochs': 25,                       
    'batch_size': 16,                   
    'learning_rate': 0.0002266,         
    'optimizer': 'adam',                
    'activation': 'relu',               
    'dropout': 0.0,                     
    'dense_units': 320,                 
    'conv_blocks': 4,                   
    'start_filters': 32,                
    'batch_norm': True,                 
    'loss': 'binary_crossentropy'
}

# --- 2. EINZEL-PARAMETER TESTS ---
SINGLE_PARAM_TESTS = {
    'batch_size': [12, 20],              
    'learning_rate': [0.00002, 0.002],  
    'dropout': [0.1, 0.3],              
    'dense_units': [280, 360],          
    'conv_blocks': [3, 5],              
    'batch_norm': [False],              
    'optimizer': ['sgd', 'rmsprop'],    
    'activation': ['elu', 'tanh']       
}

# --- 3. KOMBINATIONS-TESTS ---
COMBINATION_GROUPS = {
    'Combo_Capacity': {
        'Increase (+)': { 'dense_units': 380, 'conv_blocks': 5, 'start_filters': 48, 'dropout': 0.3 },
        'Decrease (-)': { 'dense_units': 280, 'conv_blocks': 3, 'start_filters': 20, 'dropout': 0.0 }
    },
    'Combo_Dynamics': {
        'Increase (+)': { 'batch_size': 20, 'learning_rate': 0.002 },
        'Decrease (-)': { 'batch_size': 12, 'learning_rate': 0.00002 }
    }
}

# ==========================================
# --- DATA GENERATOR (FIXIERT) ---
# ==========================================
def get_generators(batch_size):
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
    # Shuffle=False ist kritisch für die korrekte Zuordnung in den Plots!
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
# --- MODELL BAUEN (CNN) ---
# ==========================================
def create_model(params):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=INPUT_SHAPE))
    
    current_filters = params['start_filters']
    
    # Convolutional Blocks
    for i in range(params['conv_blocks']):
        model.add(layers.Conv2D(current_filters, (3, 3), padding='same'))
        
        if params.get('batch_norm', False): 
            model.add(layers.BatchNormalization())
            
        model.add(layers.Activation(params['activation']))
        model.add(layers.MaxPooling2D((2, 2)))
        current_filters *= 2
    
    model.add(layers.Flatten())
    
    # Dense Part
    model.add(layers.Dense(params['dense_units']))
    
    if params.get('batch_norm', False):
         model.add(layers.BatchNormalization())

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
# --- PLOTTING FUNKTIONEN (KORRIGIERT) ---
# ==========================================
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
BASE_COLOR = '#000000'
PLOT_PALETTE = sns.color_palette("bright", n_colors=10) 

def create_comparison_plots(results_list, category_name, save_folder):
    """
    Erstellt History, Split-Violin (Wahrscheinlichkeit) und Bar Plots.
    """
    plot_colors = [BASE_COLOR] + PLOT_PALETTE[:len(results_list)-1]

    # --- 1. HISTORY PLOT ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    for idx, res in enumerate(results_list):
        best = max(res['history']['val_accuracy'])
        label = f"{res['label']} (Best: {best:.2%})"
        style = '--' if idx == 0 else '-'
        width = 2.5 if idx == 0 else 2
        ax1.plot(res['history']['val_accuracy'], label=label, linestyle=style, linewidth=width, color=plot_colors[idx])
        
    ax1.set_title(f'Verlauf Accuracy: {category_name}', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for idx, res in enumerate(results_list):
        style = '--' if idx == 0 else '-'
        width = 2.5 if idx == 0 else 2
        ax2.plot(res['history']['val_loss'], label=res['label'], linestyle=style, linewidth=width, color=plot_colors[idx])
        
    ax2.set_title(f'Verlauf Loss: {category_name}', fontweight='bold')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'History_Comparison_{category_name}.png'), dpi=300)
    plt.close()

    # --- DATEN VORBEREITEN ---
    violin_df_list = []
    bar_data = []
    
    for res in results_list:
        bar_data.append({'Config': res['label'], 'Best Accuracy': max(res['history']['val_accuracy'])})
        
        # WICHTIG: Hier nutzen wir jetzt die Wahrscheinlichkeiten für den Plot!
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

    # --- 2. SPLIT-VIOLIN PLOT (KONSISTENT MIT EVALUATION) ---
    plt.figure(figsize=(14, 8))
    
    sns.violinplot(data=df_violin, x='Config', y='Probability', hue='True Class', 
                   split=True, inner='quart', 
                   palette={"No QR (0)": "skyblue", "QR (1)": "orange"},
                   linewidth=1.2)
    
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    
    plt.title(f'Analyse: Verteilung der Vorhersage-Sicherheit (Split by Class) - {category_name}', fontweight='bold')
    plt.ylabel('Vorhersagewahrscheinlichkeit (0.0 = No QR, 1.0 = QR)')
    plt.xlabel('Konfiguration')
    plt.ylim(-0.05, 1.05)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Wahrheit', loc='lower right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'Violin_Split_Plot_{category_name}.png'), dpi=300)
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
    plt.savefig(os.path.join(save_folder, f'Bar_Plot_Horizontal_{category_name}.png'), dpi=300)
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
        
        # Basis
        sns.heatmap(base_cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=labels, yticklabels=labels, ax=ax1, annot_kws={"size": 14})
        ax1.set_title(f"Basis (Original)\nAcc: {max(base_res['history']['val_accuracy']):.2%}", fontweight='bold')
        ax1.set_ylabel('Wahrheit')
        ax1.set_xlabel('Vorhersage')

        # Vergleich
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
    if os.path.exists(BASE_MODELS_DIR): shutil.rmtree(BASE_MODELS_DIR)
    if os.path.exists(BASE_LOGS_DIR): shutil.rmtree(BASE_LOGS_DIR)
    os.makedirs(BASE_MODELS_DIR); os.makedirs(BASE_LOGS_DIR)

    print("\n" + "="*50 + f"\n   START: {EXPERIMENT_NAME}\n" + "="*50)

    # 1. BASIS MODELL
    print("\n--- [1/3] BASIS-MODELL TRAINIEREN ---")
    base_gen_train, base_gen_val = get_generators(BASE_PARAMS['batch_size'])
    base_model = create_model(BASE_PARAMS)
    
    history_base = base_model.fit(
        base_gen_train, epochs=BASE_PARAMS['epochs'], validation_data=base_gen_val, verbose=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    
    # DATEN FÜR PLOTS GENERIEREN
    print("   Generiere Vorhersagen (Basis)...")
    base_gen_val.reset()
    base_y_pred_prob = base_model.predict(base_gen_val, verbose=0)
    base_y_true = base_gen_val.classes

    base_best_acc = max(history_base.history['val_accuracy'])
    print(f">> Basis fertig. Accuracy: {base_best_acc:.2%}")
    
    base_result = {
        'label': 'Basis (Original)',
        'history': history_base.history,
        'params': BASE_PARAMS,
        'y_pred_prob': base_y_pred_prob, # Daten für Violin Plot
        'y_true': base_y_true
    }

    # 2. & 3. TESTS
    all_tests = {}
    all_tests.update(SINGLE_PARAM_TESTS)
    all_tests.update(COMBINATION_GROUPS)

    print("\n--- PARAMETER & KOMBINATIONS TESTS ---")
    for category_name, items in all_tests.items():
        print(f"\n>>> Teste Kategorie: {category_name} <<<")
        current_model_dir = os.path.join(BASE_MODELS_DIR, f"{EXPERIMENT_NAME}_{category_name}")
        current_log_dir = os.path.join(BASE_LOGS_DIR, f"{EXPERIMENT_NAME}_{category_name}")
        os.makedirs(current_model_dir, exist_ok=True); os.makedirs(current_log_dir, exist_ok=True)
        
        results = [base_result]
        
        iterable = items.items() if isinstance(items, dict) else [(val, val) for val in items]

        for label_suffix, val_or_dict in iterable:
            if isinstance(items, dict): # Kombi
                label = label_suffix
                param_changes = val_or_dict
            else: # Einzel
                label = f"{category_name}={val_or_dict}"
                param_changes = {category_name: val_or_dict}

            print(f"   -> Szenario: {label}")
            new_params = BASE_PARAMS.copy()
            new_params.update(param_changes)
            
            t_gen, v_gen = get_generators(new_params['batch_size'])
            model = create_model(new_params)
            
            csv_path = os.path.join(current_log_dir, f"log_{label.replace('=', '_')}.csv")
            callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), CSVLogger(csv_path)]
            
            hist = model.fit(t_gen, epochs=new_params['epochs'], validation_data=v_gen, callbacks=callbacks, verbose=1)
            model.save(os.path.join(current_model_dir, f"Model_{label.replace('=', '_')}.keras"))
            
            # DATEN FÜR PLOTS GENERIEREN
            print("      Generiere Vorhersagen...")
            v_gen.reset()
            y_pred_prob = model.predict(v_gen, verbose=0)
            y_true = v_gen.classes

            results.append({
                'label': label,
                'history': hist.history,
                'params': new_params,
                'y_pred_prob': y_pred_prob, # Daten für Violin Plot
                'y_true': y_true
            })
            
        create_comparison_plots(results, category_name, current_log_dir)
        create_confusion_matrices(results, category_name, current_log_dir)
        write_evaluation_file(results, category_name, current_log_dir, base_best_acc)

    print("\n" + "="*50 + "\n   ALLE TESTS ERFOLGREICH ABGESCHLOSSEN.\n" + "="*50)

if __name__ == "__main__":
    run_full_experiment()