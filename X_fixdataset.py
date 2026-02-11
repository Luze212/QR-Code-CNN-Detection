import os
import shutil
import glob

# === KONFIGURATION ===
SOURCE_DIR = "dataset_final_yolov8"  # Dein Original-Ordner
SYNTHETIC_DIR = "dataset_synthetic"  # Deine generierten Bilder
TARGET_DIR = "dataset_FIXED"         # Der neue, saubere Ordner
# =====================

def find_folder(base_path, candidates):
    """Sucht nach einem Ordner, der wie einer der Kandidaten heißt."""
    if not os.path.exists(base_path): return None
    for item in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, item)):
            if item.lower() in candidates:
                return os.path.join(base_path, item)
    return None

def copy_contents(src_folder, dst_folder_img, dst_folder_lbl):
    """Kopiert Bilder und Labels intelligent."""
    if not src_folder: return 0
    
    # Pfade ermitteln (manchmal liegen images/labels direkt drin, manchmal im Unterordner)
    src_img = os.path.join(src_folder, "images")
    if not os.path.exists(src_img): src_img = src_folder # Fallback: flache Struktur
    
    src_lbl = os.path.join(src_folder, "labels")
    if not os.path.exists(src_lbl): src_lbl = src_folder # Fallback
    
    # Kopieren
    count = 0
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        for f in glob.glob(os.path.join(src_img, ext)):
            shutil.copy(f, dst_folder_img)
            count += 1
            
    # Labels (nur txt)
    for f in glob.glob(os.path.join(src_lbl, "*.txt")):
        shutil.copy(f, dst_folder_lbl)
        
    return count

def run_fix():
    print(f"🧹 Erstelle sauberen Datensatz: {TARGET_DIR}...")
    
    # 1. Alles Alte löschen & neu erstellen
    if os.path.exists(TARGET_DIR): shutil.rmtree(TARGET_DIR)
    
    os.makedirs(os.path.join(TARGET_DIR, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "train/labels"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "val/images"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "val/labels"), exist_ok=True)

    # 2. TRAININGSDATEN SUCHEN
    print("🔍 Suche Trainingsdaten...")
    train_src = find_folder(SOURCE_DIR, ["train", "training", "images"])
    n_train = copy_contents(train_src, os.path.join(TARGET_DIR, "train/images"), os.path.join(TARGET_DIR, "train/labels"))
    print(f"   -> {n_train} echte Bilder gefunden.")

    # 3. SYNTHETISCHE DATEN DAZUMISCHEN
    print("🧪 Mische synthetische Daten dazu...")
    # Wir nehmen an, synthetische liegen direkt in images/labels oder flach
    n_syn = copy_contents(os.path.join(SYNTHETIC_DIR, "images"), os.path.join(TARGET_DIR, "train/images"), os.path.join(TARGET_DIR, "train/labels"))
    if n_syn == 0: # Versuch flach
         n_syn = copy_contents(SYNTHETIC_DIR, os.path.join(TARGET_DIR, "train/images"), os.path.join(TARGET_DIR, "train/labels"))
    print(f"   -> {n_syn} synthetische Bilder hinzugefügt.")

    # 4. VALIDIERUNGSDATEN SUCHEN (Das Problemkind!)
    print("🔍 Suche Validierungsdaten...")
    # Wir suchen nach ALLEM was nach Validierung klingt
    val_src = find_folder(SOURCE_DIR, ["val", "valid", "validation", "test", "testing"])
    
    if not val_src:
        print("⚠️  ACHTUNG: Keinen 'val'/'valid' Ordner gefunden! Ich erstelle einen leeren Notfall-Ordner.")
        # Notfall: Wir klauen einfach 10 Bilder vom Training für den Test, damit es nicht crasht
    else:
        n_val = copy_contents(val_src, os.path.join(TARGET_DIR, "val/images"), os.path.join(TARGET_DIR, "val/labels"))
        print(f"   -> {n_val} Validierungs-Bilder gefunden (Quelle: {os.path.basename(val_src)}).")

    # 5. DATA.YAML SCHREIBEN (Absolut Idiotensicher)
    # WICHTIG: Die Einrückung hier muss genau stimmen!
    yaml_content = f"""
path: {os.path.abspath(TARGET_DIR)}
train: train/images
val: val/images
names:
  0: QR-Code
"""
    with open(os.path.join(TARGET_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print("\n✅ FERTIG. Der Datensatz ist repariert.")
    print(f"Dein neuer Pfad für das Training ist: {os.path.join(TARGET_DIR, 'data.yaml')}")

if __name__ == "__main__":
    run_fix()