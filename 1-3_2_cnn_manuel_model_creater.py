import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping # type: ignore
import matplotlib.pyplot as plt
import os
import time 

# --- KONFIGURATION PFADE ---
BASE_DATA_DIR = 'dataset_final_boxes' 
LOGS_BASE_DIR = 'logs'
MODELS_BASE_DIR = 'models'

#--- HILFSFUNKTIONEN ---
def get_user_input(prompt, default_value, value_type=str):
    """Hilfsfunktion für Nutzereingaben mit Standardwerten"""
    user_val = input(f"{prompt} [Default: {default_value}]: ").strip()
    if not user_val:
        return default_value
    try:
        return value_type(user_val)
    except ValueError:
        print(f"Ungültige Eingabe. Nutze Standardwert: {default_value}")
        return default_value

# --- MODELL ERSTELLUNG ---
def create_dynamic_model(input_shape, config):
    """Erstellt das Modell basierend auf deinen Eingaben"""
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    # --- Convolutional Blöcke (Dynamisch) ---
    current_filters = config['start_filters']
    
    for i in range(config['num_conv_blocks']):
        model.add(layers.Conv2D(current_filters, (3, 3), activation=config['activation'], padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())
        current_filters *= 2 # Filter verdoppeln für nächste Schicht
    
    # --- Classifier ---
    model.add(layers.Flatten())
    model.add(layers.Dense(config['dense_units'], activation=config['activation']))
    model.add(layers.Dropout(config['dropout']))
    
    # Output Layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# --- Hauptlogik ---
def main():
    print("="*50)
    print("MANUELLES CNN TUNING TOOL (mit Zeitmessung)")
    print("="*50)
    
    # 1. Namen festlegen
    run_name = input("Name für diesen Versuch eingeben: ").strip()
    if not run_name: run_name = "unnamed_experiment"
    
    # Ordnerstruktur anlegen
    # Log-Ordner: Hier kommen Details rein
    experiment_log_dir = os.path.join(LOGS_BASE_DIR, run_name)
    os.makedirs(experiment_log_dir, exist_ok=True)
    
    # Model-Ordner: Hier kommen NUR die .keras Dateien rein (für den Vergleich später)
    os.makedirs(MODELS_BASE_DIR, exist_ok=True)
    
    print(f"\nBitte Hyperparameter wählen (Enter für Default):")
    
    # 2. Parameter abfragen
    config = {
        'epochs': get_user_input("Anzahl Epochen (bsp. 25)", 25, int),
        'batch_size': get_user_input("Batch Size (bsp. 32)", 32, int),
        'learning_rate': get_user_input("Lernrate (bsp. 0.001)", 0.001, float),
        'optimizer_name': get_user_input("Optimierer (adam, sgd, rmsprop)", "adam", str).lower(),
        'loss_function': get_user_input("Loss Funktion (binary_crossentropy, mse)", "binary_crossentropy", str),
        'activation': get_user_input("Aktivierungsfunktion (relu, tanh, elu)", "relu", str),
        'num_conv_blocks': get_user_input("Anzahl Conv-Blöcke (Tiefe, bsp. 3)", 3, int),
        'start_filters': get_user_input("Start-Filter (Breite, bsp. 32)", 32, int),
        'dense_units': get_user_input("Neuronen im Dense Layer (bsp. 256)", 256, int),
        'dropout': get_user_input("Dropout Rate (0.0 - 1.0)", 0.5, float),
        'img_size': 300
    }

    # Parameter sichern
    with open(os.path.join(experiment_log_dir, 'config.txt'), 'w') as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
    
    # 3. Daten laden
    print("\nLade Daten...")
    if not os.path.exists(BASE_DATA_DIR):
        print(f"FEHLER: Dataset '{BASE_DATA_DIR}' nicht gefunden!")
        return

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(BASE_DATA_DIR, 'train'),
        target_size=(config['img_size'], config['img_size']),
        batch_size=config['batch_size'],
        class_mode='binary', shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(BASE_DATA_DIR, 'val'),
        target_size=(config['img_size'], config['img_size']),
        batch_size=config['batch_size'],
        class_mode='binary', shuffle=False
    )

    # 4. Modell bauen
    model = create_dynamic_model((config['img_size'], config['img_size'], 3), config)
    
    if config['optimizer_name'] == 'sgd':
        opt = optimizers.SGD(learning_rate=config['learning_rate'])
    elif config['optimizer_name'] == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=config['learning_rate'])
    else:
        opt = optimizers.Adam(learning_rate=config['learning_rate'])
        
    model.compile(loss=config['loss_function'], optimizer=opt, metrics=['accuracy'])
    
    # 5. Callbacks
    model_save_path = os.path.join(MODELS_BASE_DIR, f'{run_name}.keras')
    
    callbacks = [
        CSVLogger(os.path.join(experiment_log_dir, 'history.csv')),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]
    
    # 6. Training mit Zeitmessung
    print(f"\nStarte Training: {run_name}")
    start_time = time.time() # Startuhr
    
    history = model.fit(
        train_gen,
        epochs=config['epochs'],
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    end_time = time.time() # Stoppuhr
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60
    
    # 7. Ergebnisse
    best_val_acc = max(history.history['val_accuracy'])
    
    # Report erstellen (Zeit & Ergebnis)
    report_path = os.path.join(experiment_log_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"REPORT für {run_name}\n")
        f.write("========================\n")
        f.write(f"Dauer (Sekunden): {duration_seconds:.2f} s\n")
        f.write(f"Dauer (Minuten):  {duration_minutes:.2f} min\n")
        f.write(f"Beste Val Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Gestoppt nach Epoche: {len(history.history['accuracy'])}\n")
    
    # Plotting
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training')
    plt.plot(epochs_range, val_acc, label='Validation')
    plt.title(f'Accuracy (Best: {best_val_acc:.2%})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training')
    plt.plot(epochs_range, val_loss, label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plot_path = os.path.join(experiment_log_dir, 'training_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    print("\nVersuch abgeschlossen.")
    print(f"Trainingszeit: {duration_minutes:.2f} Minuten")
    print(f"Modell liegt in: {model_save_path}")
    print(f"Logs & Report in: {experiment_log_dir}")

if __name__ == "__main__":
    main()