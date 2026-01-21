import pandas as pd
import matplotlib.pyplot as plt
import os

# --- KONFIGURATION ---
# Pfad zu deinen Log-Dateien für die Plot erstellt werden soll
LOG_FILE_CSV = 'logs/tuned_cnn_Hyperband_1_300,300/best_tuned_log.csv'
NAME_PLOT_FILE = 'Bayesian_Plot.png'
TITLE = 'Bayesian'

# Wo sollen die neuen Bilder hin?
OUTPUT_DIR = 'logs/replotted_plots'

def create_standard_plot(csv_path, save_name, title_prefix):
    """Erstellt ein standardisiertes 2-Spalten-Layout (Acc & Loss)"""
    
    if not os.path.exists(csv_path):
        print(f"Datei nicht gefunden: {csv_path}")
        return

    # Daten laden
    data = pd.read_csv(csv_path)
    
    epochs = range(len(data))
    acc = data['accuracy']
    val_acc = data['val_accuracy']
    loss = data['loss']
    val_loss = data['val_loss']
    
    best_val_acc = val_acc.max()

    # Plot erstellen
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Linke Seite: Accuracy ---
    ax1.plot(epochs, acc, label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
    ax1.set_title(f'{title_prefix}: Accuracy (Best: {best_val_acc:.2%})', fontsize=14)
    ax1.set_xlabel('Epochen')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # --- Rechte Seite: Loss ---
    ax2.plot(epochs, loss, label='Training Loss', linewidth=2, color='red')
    ax2.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='orange')
    ax2.set_title(f'{title_prefix}: Loss', fontsize=14)
    ax2.set_xlabel('Epochen')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Layout straffen und speichern
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300) # Hohe Auflösung für die Arbeit
    plt.close()
    
    print(f"Plot gespeichert: {save_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generiere einheitliche Plots...")
    
    # 1. Plot neu machen
    create_standard_plot(LOG_FILE_CSV, NAME_PLOT_FILE, TITLE)

if __name__ == "__main__":
    main()