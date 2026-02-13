import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import os
import numpy as np

# ==============================================================================
# KONFIGURATION
# ==============================================================================

# 1. Pfad zu deinem Trainings-Ordner (Dort wo results.csv und weights/ liegen)
TRAIN_RUN_DIR = "runs/detect/QR_Code_1280s/train_M4_HighRes_SmartAug" 

OUTPUT_DIR = "logs/yolo_evaluation_plots"

# 3. Styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

# ==============================================================================

def plot_training_curves(run_dir, save_dir):
    csv_path = os.path.join(run_dir, "results.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ Fehler: Keine results.csv gefunden in {csv_path}")
        return

    # Daten laden und Spaltennamen säubern (YOLO hat oft Leerzeichen)
    df = pd.read_csv(csv_path)
    df.columns = [x.strip() for x in df.columns]

    epochs = df['epoch']
    
    # Setup Plot (2 Bilder nebeneinander: Loss & Accuracy/mAP)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT 1: LOSS ---
    # Wir plotten Box Loss (Wie gut sitzt der Rahmen?) und Class Loss (Ist es ein QR-Code?)
    # Train
    ax1.plot(epochs, df['train/box_loss'], label='Train Box Loss', color='tab:blue', linewidth=2)
    ax1.plot(epochs, df['train/cls_loss'], label='Train Class Loss', color='tab:orange', linewidth=2)
    # Val
    ax1.plot(epochs, df['val/box_loss'], label='Val Box Loss', color='tab:blue', linestyle='--', linewidth=2, alpha=0.7)
    ax1.plot(epochs, df['val/cls_loss'], label='Val Class Loss', color='tab:orange', linestyle='--', linewidth=2, alpha=0.7)
    
    ax1.set_title("Lernkurve: Fehler (Loss)", fontweight='bold')
    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)

    # --- PLOT 2: ACCURACY (mAP) ---
    # mAP50 ist quasi die "Accuracy" bei Objekterkennung (IoU > 0.5)
    # Precision und Recall zeigen wir auch leicht transparent an
    ax2.plot(epochs, df['metrics/mAP50(B)'], label='mAP@50 (Genauigkeit)', color='tab:green', linewidth=3)
    ax2.plot(epochs, df['metrics/precision(B)'], label='Precision', color='tab:red', linestyle=':', alpha=0.6)
    ax2.plot(epochs, df['metrics/recall(B)'], label='Recall', color='tab:purple', linestyle=':', alpha=0.6)

    ax2.set_title("Performance: mAP & Metriken", fontweight='bold')
    ax2.set_xlabel("Epoche")
    ax2.set_ylabel("Score (0.0 - 1.0)")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "custom_training_curves.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Verlaufskurven gespeichert: {save_path}")
    plt.close()

def plot_confusion_matrix(run_dir, save_dir):
    weights_path = os.path.join(run_dir, "weights", "best.pt")
    
    if not os.path.exists(weights_path):
        print(f"❌ Fehler: Kein Modell gefunden in {weights_path}")
        return

    print("🔄 Lade Modell und führe Validierung durch (für Confusion Matrix)...")
    # Modell laden
    model = YOLO(weights_path)
    
    # Validierung laufen lassen (auf dem Validierungs-Set aus der data.yaml)
    # plots=False, damit er nicht die Standard-Bilder überschreibt
    metrics = model.val(plots=False, verbose=False)
    
    # Die Confusion Matrix extrahieren
    # YOLO speichert die Matrix intern. Wir müssen sie normalisieren.
    cm = metrics.confusion_matrix.matrix
    
    # Klassen-Namen (sollte ['QR-Code', 'background'] sein)
    # Achtung: YOLO fügt 'background' oft implizit hinzu als letzte Zeile/Spalte
    names = list(model.names.values()) 
    if len(names) < cm.shape[0]:
        names.append("Hintergrund")

    # Plotten mit Seaborn
    plt.figure(figsize=(10, 8))
    
    # Normalisieren (damit wir % sehen) - Optional, aber besser verständlich
    # Wir addieren eine winzige Zahl (1e-9), um Division durch Null zu vermeiden
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', 
                xticklabels=names, yticklabels=names,
                annot_kws={"size": 14, "weight": "bold"}, cbar=False)
    
    plt.title('Confusion Matrix (Normalisiert)', fontsize=16, fontweight='bold')
    plt.ylabel('Wahre Klasse (Ground Truth)', fontsize=12)
    plt.xlabel('Vorhergesagte Klasse (Prediction)', fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "custom_confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Confusion Matrix gespeichert: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Ordner erstellen
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- Starte Auswertung für: {TRAIN_RUN_DIR} ---")
    
    # 1. Kurven
    plot_training_curves(TRAIN_RUN_DIR, OUTPUT_DIR)
    
    # 2. Matrix
    plot_confusion_matrix(TRAIN_RUN_DIR, OUTPUT_DIR)
    
    print("\n🎉 Fertig! Schau in den Ordner 'evaluation_plots'.")