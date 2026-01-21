import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- KONFIGURATION ---
SOURCE_DIR = "./dataset_final_yolov8"  # Dataset (Yolov8 - als Polygone gelabelt)
TARGET_DIR = "./dataset_final_boxes" # Ziet-Dataset (Box-Format für CNN)
IMG_SIZE = 300 
TEST_SPLIT = 0.2
SEED = 42

# --- Ordner erstellen ---
def setup_directories():
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR) # Alten Ordner löschen für sauberen Start
    for split in ['train', 'val']:
        for cls in ['qr_code', 'no_qr_code']:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

# --- Koordinaten in Polygone umwandeln ---
def parse_polygons(txt_path, w, h):
    """Liest YOLO Koordinaten und wandelt sie in Pixel-Polygone um."""
    polygons = []
    if not os.path.exists(txt_path):
        return polygons
    with open(txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            # Format: class x1 y1 x2 y2 ... (normalisiert 0..1)
            coords = parts[1:]
            points = []
            for i in range(0, len(coords), 2):
                px = int(coords[i] * w)
                py = int(coords[i+1] * h)
                points.append([px, py])
            polygons.append(np.array(points, dtype=np.int32))
    return polygons

# --- Bilder ohne QR-Code erzeugen ---
def create_background_crop(img, polygons, filename, save_dir):
    """Erzeugt ein Bild OHNE QR-Code aus dem Hintergrund."""
    h, w, _ = img.shape
    # Maske: Weiß wo QR-Code ist, Schwarz wo Hintergrund ist
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, polygons, 255)
    
    # Kernel für Dilatation, um Sicherheitsabstand zum QR-Code zu haben
    kernel = np.ones((20,20), np.uint8) 
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Versuche max 50 mal einen sauberen Crop zu finden
    for i in range(50):
        x = random.randint(0, max(0, w - IMG_SIZE))
        y = random.randint(0, max(0, h - IMG_SIZE))
        
        # Prüfen ob im Masken-Ausschnitt weiße Pixel sind
        roi = mask[y:y+IMG_SIZE, x:x+IMG_SIZE]
        if np.sum(roi) == 0: 
            crop = img[y:y+IMG_SIZE, x:x+IMG_SIZE]
            # Prüfen ob Crop groß genug ist
            if crop.shape[0] == IMG_SIZE and crop.shape[1] == IMG_SIZE:
                save_path = os.path.join(save_dir, f"bg_{filename}")
                cv2.imwrite(save_path, crop)
                return True
    return False

# --- Haupt-Ausführungslogik ---
def main():
    print("🚀 Starte Datenverarbeitung...")
    setup_directories()
    
    # Bilder finden
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(SOURCE_DIR).rglob(ext))
    
    random.seed(SEED)
    random.shuffle(image_files)
    
    stats = {'qr_train': 0, 'qr_val': 0, 'no_qr_orig': 0, 'no_qr_gen': 0}
    
    print(f"Verarbeite {len(image_files)} Bilder...")
    
    for img_path in tqdm(image_files):
        # Label Pfad suchen
        txt_name = img_path.with_suffix('.txt').name
        potential_paths = [
            img_path.with_suffix('.txt'), # Gleicher Ordner
            img_path.parent.parent / 'labels' / txt_name # YOLO Standard
        ]
        
        label_path = None
        for p in potential_paths:
            if p.exists():
                label_path = p
                break
        
        has_qr = label_path and label_path.stat().st_size > 0
        
        # Split Zuweisung
        split = "val" if random.random() < TEST_SPLIT else "train"
        
        if has_qr:
            # 1. Originalbild -> qr_code Ordner
            dest = os.path.join(TARGET_DIR, split, "qr_code", img_path.name)
            shutil.copy(img_path, dest)
            stats[f'qr_{split}'] += 1
            
            # 2. Hintergrund ausschneiden -> no_qr_code Ordner (nur für Train)
            if split == "train":
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w, _ = img.shape
                    polys = parse_polygons(label_path, w, h)
                    dest_no = os.path.join(TARGET_DIR, "train", "no_qr_code")
                    
                    # 1-2 Crops pro Bild, um auf die Menge zu kommen
                    if create_background_crop(img, polys, img_path.name, dest_no):
                        stats['no_qr_gen'] += 1
                        
        else:
            # Bild ohne QR -> no_qr_code Ordner
            dest = os.path.join(TARGET_DIR, split, "no_qr_code", img_path.name)
            shutil.copy(img_path, dest)
            stats['no_qr_orig'] += 1

    print("\n Fertig!")
    print(f"Training QR-Codes: {stats['qr_train']}")
    print(f"Training Kein-QR (Original): ~{int(stats['no_qr_orig'] * (1-TEST_SPLIT))}")
    print(f"Training Kein-QR (Generiert): {stats['no_qr_gen']}")
    print(f"Validierung QR: {stats['qr_val']}")
    
    total_neg_train = int(stats['no_qr_orig'] * (1-TEST_SPLIT)) + stats['no_qr_gen']
    print(f"\nVerhältnis im Training: {stats['qr_train']} (QR) vs {total_neg_train} (Kein QR)")

if __name__ == "__main__":
    main()