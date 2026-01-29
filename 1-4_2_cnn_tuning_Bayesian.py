import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
import keras_tuner as kt
import matplotlib.pyplot as plt
import os

# --- KONFIGURATION ---
BASE_DIR = 'dataset_final_boxes' 
TUNED_DIR = 'models'      
LOG_DIR = 'logs/tuned_cnn_Bayesian_2_256,256' 

IMG_SIZE = (256, 256)

# Tuning Einstellungen
MAX_TRIALS = 20           # 20 gezielte Versuche (Bayesian ist langsam aber schlau)
EXECUTION_PER_TRIAL = 1   # Jeden Versuch 1x durchführen
EPOCHS_TUNING = 20        # Genug Zeit geben
EPOCHS_FINAL = 35         # Finales Training länger
project='tuning_Bayesian_2_256,256'    # Projektname für tuning_results
PLOT_TITEL = 'Bayesian'    # Titel für den Plot

# Ordner erstellen
os.makedirs(TUNED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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

class QRCodeHyperModel(kt.HyperModel):
    def build(self, hp):
        model = models.Sequential()
        model.add(layers.Input(shape=IMG_SIZE + (3,)))
        
        # --- TUNING BEREICH 1: Faltung (Das Auge) ---
        # Wir testen, ob wir mit 32 oder 64 Filtern starten sollen
        start_filters = hp.Choice('start_filters', [32, 64])
        
        # Anzahl der Blöcke: 3 ist Standard, testen wir mal 4
        num_blocks = hp.Int('num_blocks', 3, 4)
        
        current_filters = start_filters
        for i in range(num_blocks):
            model.add(layers.Conv2D(current_filters, (3, 3), activation='relu', padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))
            
            # Batch Norm ist fast immer gut, aber wir lassen den Tuner entscheiden
            if hp.Boolean('batch_norm'):
                model.add(layers.BatchNormalization())
            
            # Filter verdoppeln (typische CNN Struktur)
            current_filters *= 2
        
        model.add(layers.Flatten())
        
        # --- TUNING BEREICH 2: Das Gehirn (Dense) ---
        model.add(layers.Dense(
            units=hp.Int('dense_units', 128, 512, step=64),
            activation='relu'
        ))
        
        # WICHTIG: Wir zwingen den Tuner in den niedrigen Bereich (0.0 bis 0.35)
        # Denn wir wissen ja schon, dass 0.2 gut war.
        model.add(layers.Dropout(
            rate=hp.Float('dropout', 0.0, 0.35, step=0.05)
        ))
        
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # --- TUNING BEREICH 3: Lernen ---
        # Learning Rate feiner tunen
        lr = hp.Float('learning_rate', 1e-4, 5e-3, sampling='log')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice('batch_size', [16, 32]) # 64 oft zu groß für Memory
        train_gen, val_gen = get_data(batch_size)
        
        return model.fit(
            train_gen, 
            validation_data=val_gen, 
            *args, 
            **kwargs
        )

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



def main():
    # 1. Der "Sniper" Tuner (BayesianOptimization)
    tuner = kt.BayesianOptimization(
        QRCodeHyperModel(),
        objective='val_accuracy',
        max_trials=MAX_TRIALS,
        num_initial_points=5, # Erst 5 zufällig, dann 15 schlau
        directory='tuning_results',
        project_name=project, # Neuer Name!
        overwrite=True # Löscht alten Cache dieses Projekts
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print(f"Starte Bayesian-Tuning ({MAX_TRIALS} Trials)...")
    print("Strategie: Bayesian Optimization")
    
    tuner.search(epochs=EPOCHS_TUNING, callbacks=[early_stop])
    
    # --- AUSWERTUNG ---
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\nGEWINNER GEFUNDEN!")
    print(f"Start Filters: {best_hps.get('start_filters')}")
    print(f"Num Blocks: {best_hps.get('num_blocks')}")
    print(f"Dropout: {best_hps.get('dropout')}")
    print(f"Dense Units: {best_hps.get('dense_units')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    
    # Speichern der Parameter
    with open(os.path.join(LOG_DIR, 'best_bayesianparams.txt'), 'w') as f:
        for p, v in best_hps.values.items():
            f.write(f"{p}: {v}\n")

    # --- FINALES TRAINING ---
    print("\nTrainiere finales Modell mit Bayesian-Parametern...")
    model = tuner.hypermodel.build(best_hps)
    best_batch_size = best_hps.get('batch_size')
    train_gen, val_gen = get_data(best_batch_size)
    
    callbacks = [
        CSVLogger(os.path.join(LOG_DIR, 'best_tuned_log.csv')),
        ModelCheckpoint(os.path.join(TUNED_DIR, 'best_tuned_model_Bayesian.keras'), 
                        monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        # Extra-Trick: Wenn es hakt, Lernrate verringern
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS_FINAL,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # Plotten
    plot_and_save_history(history, LOG_DIR, 'best_tuned_plot.png', PLOT_TITEL)
    
    best_final_acc = max(history.history['val_accuracy'])
    
    print(f"\nFERTIG! Bayesian Accuracy: {best_final_acc:.4f}")
    
if __name__ == "__main__":
    main()