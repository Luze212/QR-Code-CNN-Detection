import os
import shutil
import glob

# ==============================================================================
# KONFIGURATION
# ==============================================================================

# 1. Dein originaler, sauberer Datensatz
ORIGINAL_DATASET_DIR = "dataset_final_yolov8"

# 2. Die neuen synthetischen Bilder
SYNTHETIC_DATASET_DIR = "dataset_synthetic"

# 3. Der neue Ordner, der erstellt wird (wird bei jedem Start überschrieben!)
NEW_DATASET_DIR = "dataset_combined_v1"

# ==============================================================================

def copy_files(src_pattern, dest_dir):
    """Hilfsfunktion zum Kopieren von Dateien"""
    files = glob.glob(src_pattern)
    if not files:
        return 0
    
    for f in files:
        shutil.copy(f, dest_dir)
    return len(files)

def create_dataset():
    print(f"🚀 Erstelle neuen Datensatz: {NEW_DATASET_DIR}")

    # 1. Alten Combined-Ordner löschen (damit wir sauber starten)
    if os.path.exists(NEW_DATASET_DIR):
        print("   -> Lösche alte Version...")
        shutil.rmtree(NEW_DATASET_DIR)
    
    # 2. Ordnerstruktur erstellen
    subdirs = ["train/images", "train/labels", "val/images", "val/labels"]
    for sd in subdirs:
        os.makedirs(os.path.join(NEW_DATASET_DIR, sd), exist_ok=True)

    print("------------------------------------------------")

    # 3. ORIGINALE TRAININGS-DATEN KOPIEREN
    print("1️⃣  Kopiere Originale Trainingsdaten...")
    n_img = copy_files(os.path.join(ORIGINAL_DATASET_DIR, "train/images/*"), os.path.join(NEW_DATASET_DIR, "train/images"))
    n_lbl = copy_files(os.path.join(ORIGINAL_DATASET_DIR, "train/labels/*"), os.path.join(NEW_DATASET_DIR, "train/labels"))
    print(f"   -> {n_img} Bilder und {n_lbl} Labels kopiert.")

    # 4. SYNTHETISCHE DATEN DAZUMISCHEN (NUR TRAIN!)
    print("2️⃣  Mische synthetische Daten dazu...")
    # Achtung: Pfadstruktur von dataset_synthetic beachten (meist direkt images/labels)
    # Falls dein synthetic folder auch 'train' hat, pass den Pfad unten an.
    # Wir gehen davon aus: dataset_synthetic/images/*.jpg
    
    # Checken, ob Unterordner 'train' existiert oder ob sie direkt liegen
    syn_source_img = os.path.join(SYNTHETIC_DATASET_DIR, "images")
    if not os.path.exists(syn_source_img): 
        # Fallback falls dein Generator keine Unterordner gemacht hat
        syn_source_img = SYNTHETIC_DATASET_DIR 
        syn_source_lbl = SYNTHETIC_DATASET_DIR
    else:
        syn_source_lbl = os.path.join(SYNTHETIC_DATASET_DIR, "labels")

    n_syn_img = copy_files(os.path.join(syn_source_img, "*.jpg"), os.path.join(NEW_DATASET_DIR, "train/images"))
    # Auch png versuchen falls gemischt
    n_syn_img += copy_files(os.path.join(syn_source_img, "*.png"), os.path.join(NEW_DATASET_DIR, "train/images"))
    n_syn_lbl = copy_files(os.path.join(syn_source_lbl, "*.txt"), os.path.join(NEW_DATASET_DIR, "train/labels"))
    print(f"   -> {n_syn_img} synthetische Bilder hinzugefügt.")

    # 5. VALIDIERUNGS-DATEN KOPIEREN (WICHTIG: NUR ORIGINALE!)
    # Wir wollen ja auf echten Daten testen, nicht auf Fake-Daten.
    print("3️⃣  Kopiere Validierungsdaten (nur echte)...")
    n_val_img = copy_files(os.path.join(ORIGINAL_DATASET_DIR, "val/images/*"), os.path.join(NEW_DATASET_DIR, "val/images"))
    n_val_lbl = copy_files(os.path.join(ORIGINAL_DATASET_DIR, "val/labels/*"), os.path.join(NEW_DATASET_DIR, "val/labels"))
    print(f"   -> {n_val_img} Validierungs-Bilder übertragen.")

    # 6. NEUE DATA.YAML ERSTELLEN
    print("4️⃣  Erstelle data.yaml...")
    yaml_content = f"""
path: {os.path.abspath(NEW_DATASET_DIR)}
train: train/images
val: val/images

# Classes
names:
  0: QR-Code
"""
    with open(os.path.join(NEW_DATASET_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print("------------------------------------------------")
    print("✅ FERTIG! Dein neuer Datensatz liegt in:")
    print(f"   {NEW_DATASET_DIR}")
    print("   Die 'data.yaml' liegt darin bereit.")

if __name__ == "__main__":
    create_dataset()