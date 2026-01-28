import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
import keras_tuner as kt
import matplotlib.pyplot as plt
import os
import gc

# --- KONFIGURATION ---
BASE_DIR = 'dataset_final_boxes' 
TUNED_DIR = 'models'      
LOG_DIR = 'logs/tuned_cnn_Bayesian_Balanced' 

# Wir bleiben bei der Sieger-Auflösung
IMG_SIZE = (256, 256) 

# Tuning Einstellungen
MAX_TRIALS = 15           # Etwas weniger Trials reichen oft
EXECUTION_PER_TRIAL = 1   
EPOCHS_TUNING = 20        
EPOCHS_FINAL = 35         
project = 'tuning_Bayesian_Balanced'    
PLOT_TITEL = 'Balanced Bayesian (Acc & Precision)'   

# Ordner erstellen
os.makedirs(TUNED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- METRIKEN ---
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'), 
    tf.keras.metrics.Recall(name='recall')
]

def get_data(batch_size):
    """Dynamischer Generator basierend auf Batch Size"""
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
        os.path.join(BASE_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return train_gen, val_gen

class QRCodeBalancedHyperModel(kt.HyperModel):
    def build(self, hp):
        model = models.Sequential()
        model.add(layers.Input(shape=IMG_SIZE + (3,)))
        
        # --- L2 Regularisierung (Deutlich abgeschwächt) ---
        # Wir testen "keine" oder "ganz wenig", um das Signal nicht zu töten
        l2_rate = hp.Choice('l2_rate', [0.0, 1e-5, 1e-4]) 
        
        # --- CONV BLÖCKE ---
        start_filters = hp.Choice('start_filters', [32, 64])
        num_blocks = hp.Int('num_blocks', 3, 4)
        
        current_filters = start_filters
        for i in range(num_blocks):
            # Wir wenden L2 nur auf die Kernel an, sehr sanft
            if l2_rate > 0:
                reg = regularizers.l2(l2_rate)
            else:
                reg = None
                
            model.add(layers.Conv2D(
                current_filters, (3, 3), 
                activation='relu', 
                padding='same',
                kernel_regularizer=reg
            ))
            model.add(layers.MaxPooling2D((2, 2)))
            
            if hp.Boolean('batch_norm'):
                model.add(layers.BatchNormalization())
            
            current_filters *= 2
        
        model.add(layers.Flatten())
        
        # --- DENSE LAYERS ---
        model.add(layers.Dense(
            units=hp.Int('dense_units', 128, 512, step=64),
            activation='relu',
            kernel_regularizer=reg # Gleiche Regularisierung wie oben
        ))
        
        # --- DROPOUT (Der wichtigste Hebel gegen False Positives) ---
        # Wir testen moderate bis hohe Werte. 
        # Nicht zu hoch (0.6 war evtl. zu viel), nicht zu niedrig.
        model.add(layers.Dropout(
            rate=hp.Float('dropout', 0.25, 0.5, step=0.05)
        ))
        
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Learning Rate
        lr = hp.Float('learning_rate', 1e-4, 2e-3, sampling='log')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=METRICS
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        tf.keras.backend.clear_session()
        gc.collect()
        
        batch_size = hp.Choice('batch_size', [16, 32]) 
        train_gen, val_gen = get_data(batch_size)
        
        return model.fit(train_gen, validation_data=val_gen, *args, **kwargs)

# --- PLOT FUNKTION (Dein Standard) ---
def plot_and_save_history(history, folder, filename_prefix, title_prefix):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    best_val_acc = max(val_acc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy
    ax1.plot(epochs, acc, label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
    ax1.set_title(f'{title_prefix}: Accuracy (Best: {best_val_acc:.2%})', fontsize=14)
    ax1.set_xlabel('Epochen')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Loss
    ax2.plot(epochs, loss, label='Training Loss', linewidth=2, color='red')
    ax2.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='orange')
    ax2.set_title(f'{title_prefix}: Loss', fontsize=14)
    ax2.set_xlabel('Epochen')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(folder, f'{filename_prefix}_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Grafik gespeichert: {plot_path}")

def main():
    # 1. TUNER SETUP
    tuner = kt.BayesianOptimization(
        QRCodeBalancedHyperModel(),
        # WICHTIG: Zurück zu Accuracy! 
        # Precision-Optimierung ist zu gefährlich für die Erkennungsrate.
        # False Positives bekämpfen wir durch Dropout und Architektur.
        objective='val_accuracy', 
        max_trials=MAX_TRIALS,
        num_initial_points=5,
        directory='tuning_results',
        project_name=project,
        overwrite=True
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print(f"🚀 Starte BALANCED Bayesian-Tuning...")
    
    tuner.search(epochs=EPOCHS_TUNING, callbacks=[early_stop])
    
    # --- AUSWERTUNG ---
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n🎯 GEWINNER GEFUNDEN!")
    print(f"L2 Rate: {best_hps.get('l2_rate')}")
    print(f"Dropout: {best_hps.get('dropout')}")
    print(f"Filters: {best_hps.get('start_filters')}")
    print(f"LR: {best_hps.get('learning_rate')}")
    
    with open(os.path.join(LOG_DIR, 'best_balanced_params.txt'), 'w') as f:
        for p, v in best_hps.values.items():
            f.write(f"{p}: {v}\n")

    # --- FINALES TRAINING ---
    print("\n🏋️‍♂️ Trainiere finales Modell...")
    model = tuner.hypermodel.build(best_hps)
    best_batch_size = best_hps.get('batch_size')
    train_gen, val_gen = get_data(best_batch_size)
    
    callbacks = [
        CSVLogger(os.path.join(LOG_DIR, 'balanced_log.csv')),
        # Wir speichern basierend auf Accuracy, schauen aber später auf Precision
        ModelCheckpoint(os.path.join(TUNED_DIR, 'best_tuned_model_Balanced.keras'), 
                        monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS_FINAL,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    plot_and_save_history(history, LOG_DIR, 'balanced_tuned', PLOT_TITEL)
    
    print("\n✅ FERTIG! Finale Werte:")
    print(f"Accuracy: {max(history.history['val_accuracy']):.4f}")
    # Diese beiden Werte sind jetzt entscheidend für dich:
    print(f"Precision: {max(history.history['val_precision']):.4f}") 
    print(f"Recall: {max(history.history['val_recall']):.4f}")

if __name__ == "__main__":
    main()