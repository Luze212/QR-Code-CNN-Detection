import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint # type: ignore
import keras_tuner as kt
import matplotlib.pyplot as plt
import os

# --- KONFIGURATION ---
BASE_DIR = 'dataset_final_boxes' # Dataset
TUNED_DIR = 'models'      # Zielordner für Ergebnisse
LOG_DIR = 'logs/tuned_cnn_Hyperband_2_256,256'  # Log-Ordner
IMG_SIZE = (256, 256)
MAX_EPOCHS_TUNING = 15          # Epochen Hyperparameter-Tuning
MAX_EPOCHS_FINAL = 30           # Epochen finale Model´ß
project='tuning_Hyperband_2_256,256'
PLOT_TITEL = 'Hyperband'    # Titel für den Plot
# Ordner erstellen
os.makedirs(TUNED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Daten und Argumentation ---
def get_data():
    """Liefert die Generatoren (Light Augmentation als Basis)"""
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
    return train_datagen, val_datagen

# --- Hypermodel Definition ---
class QRCodeHyperModel(kt.HyperModel):
    def build(self, hp):
        model = models.Sequential()
        model.add(layers.Input(shape=IMG_SIZE + (3,)))
        
        # Tuning: Conv Blöcke
        num_conv_blocks = hp.Int('num_conv_blocks', 1, 3, default=1)
        act_func = hp.Choice('activation', ['relu', 'elu', 'tanh'])
        
        for i in range(num_conv_blocks):
            filters = hp.Int(f'filters_{i}', 32, 128, step=32)
            model.add(layers.Conv2D(filters, (3, 3), activation=act_func, padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))
            if hp.Boolean('use_batch_norm'):
                model.add(layers.BatchNormalization())
        
        model.add(layers.Flatten())
        
        # Tuning: Dense Layer & Dropout
        model.add(layers.Dense(
            units=hp.Int('dense_units', 64, 512, step=64),
            activation=act_func
        ))
        
        model.add(layers.Dropout(
            rate=hp.Float('dropout', 0.2, 0.6, step=0.1)
        ))
        
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Tuning: Optimizer & LR
        lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        opt_choice = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
        
        if opt_choice == 'adam': optimizer = optimizers.Adam(learning_rate=lr)
        elif opt_choice == 'sgd': optimizer = optimizers.SGD(learning_rate=lr)
        else: optimizer = optimizers.RMSprop(learning_rate=lr)
            
        loss_choice = hp.Choice('loss_fn', ['binary_crossentropy', 'mse'])
        
        model.compile(optimizer=optimizer, loss=loss_choice, metrics=['accuracy'])
        return model

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice('batch_size', [16, 32, 64])
        train_datagen, val_datagen = get_data()
        
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
        return model.fit(train_gen, validation_data=val_gen, *args, **kwargs)

# --- Erstellung von Plots ---
def plot_and_save_history(history, folder, filename_prefix, title_prefix):
    """
    Erstellt einen Plot im standardisierten Design (identisch zu Replot_function).
    Zeigt Accuracy (mit Bestwert) und Loss nebeneinander an.
    """
    # Daten aus dem History-Objekt extrahieren
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    # Bestwert ermitteln für den Titel
    best_val_acc = max(val_acc)

    # Plot erstellen (Gleiche Größe wie dein Referenz-Plot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Linke Seite: Accuracy ---
    ax1.plot(epochs, acc, label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
    # Titel mit Bestwert-Anzeige
    ax1.set_title(f'{title_prefix}: Accuracy (Best: {best_val_acc:.2%})', fontsize=14)
    ax1.set_xlabel('Epochen')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # --- Rechte Seite: Loss ---
    # Hier nutzen wir Rot/Orange für bessere Unterscheidung
    ax2.plot(epochs, loss, label='Training Loss', linewidth=2, color='red')
    ax2.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='orange')
    ax2.set_title(f'{title_prefix}: Loss', fontsize=14)
    ax2.set_xlabel('Epochen')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Layout straffen und speichern
    plt.tight_layout()
    
    # Pfad zusammenbauen
    plot_path = os.path.join(folder, f'{filename_prefix}_plot.png')
    
    # Speichern mit hoher Auflösung (300 DPI)
    plt.savefig(plot_path, dpi=300)
    plt.close()

# --- Haupt-Ausführungslogik ---
def main():
    # 1. Tuning Setup
    tuner = kt.Hyperband(
        QRCodeHyperModel(),
        objective='val_accuracy',
        max_epochs=MAX_EPOCHS_TUNING,
        factor=3,
        directory='tuning_results',
        project_name=project
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=4)
    
    print("PHASE 1: Suche nach besten Hyperparametern...")
    tuner.search(callbacks=[early_stop])
    
    # 2. Beste Parameter 
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_batch_size = best_hps.get('batch_size')
    
    # Parameter in Textdatei speichern
    params_path = os.path.join(LOG_DIR, 'best_hyperparameters.txt')
    with open(params_path, 'w') as f:
        f.write("BESTE HYPERPARAMETER:\n")
        f.write("=====================\n")
        for param, value in best_hps.values.items():
            f.write(f"{param}: {value}\n")
    print(f"Parameter-Log gespeichert: {params_path}")

    # 3. Finales Training mit besten Parametern
    print("\nPHASE 2: Finales Training mit Besten Parametern...")
    
    # Modell bauen
    model = tuner.hypermodel.build(best_hps)
    
    # Daten laden entsprechend gefundenen Parametern
    train_datagen, val_datagen = get_data()
    train_gen = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=best_batch_size,
        class_mode='binary',
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=best_batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    # Callbacks für das finale Training
    log_path = os.path.join(LOG_DIR, 'best_tuned_log.csv')
    model_path = os.path.join(TUNED_DIR, 'best_tuned_model_Hyperband.keras')
    
    callbacks = [
        CSVLogger(log_path, append=False),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]
    
    # Start
    history = model.fit(
        train_gen,
        epochs=MAX_EPOCHS_FINAL,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # 4. Plotten
    plot_and_save_history(history, LOG_DIR, 'best_tuned_plot.png', PLOT_TITEL)
    
    print("\nFERTIG! Alle Dateien liegen in models/tuned/:")
    print(f"- Modell: {model_path}")
    print(f"- Log (CSV): {log_path}")
    print(f"- Plot (PNG): best_tuned_plot.png")
    print(f"- Params (TXT): best_hyperparameters.txt")

if __name__ == "__main__":
    main()