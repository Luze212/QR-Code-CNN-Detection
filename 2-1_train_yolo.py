from ultralytics import YOLO
import os
import yaml

# ================= KONFIGURATION =================
DATASET_DIR = os.path.abspath('dataset_small_yolov8')
DATA_YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

PROJECT_NAME = 'QR_Code_1280s'
EPOCHS = 80 # Etwas mehr Epochen, da wir stark augmentieren
IMG_SIZE = 1280 # DAS ist der Schlüssel für QR-Codes!

# Hardware M4 (24GB+):
BATCH_SIZE = 8 
WORKERS = 4  # Der M4 packt locker 4 Worker für Dataloading
MODEL_NAME = "yolov8s.pt" 
# =================================================

def check_and_fix_yaml():
    if not os.path.exists(DATA_YAML_PATH):
        return False
    with open(DATA_YAML_PATH, 'r') as f:
        data = yaml.safe_load(f)
    data['train'] = os.path.join(DATASET_DIR, 'train', 'images')
    data['val']   = os.path.join(DATASET_DIR, 'valid', 'images')
    data['test']  = os.path.join(DATASET_DIR, 'test', 'images')
    data['nc'] = 1
    data['names'] = ['QR-Code']
    with open(DATA_YAML_PATH, 'w') as f:
        yaml.dump(data, f)
    return True

def run_training():
    if not check_and_fix_yaml():
        return

    print(f"\nStarte High-Res Training auf M4 mit {MODEL_NAME}...")

    model = YOLO(MODEL_NAME)

    results = model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS, 
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            workers=WORKERS,      
            project=PROJECT_NAME,
            name="train_M4_HighRes_SmartAug", # Neuer Name
            plots=True,
            exist_ok=True,
            device='mps',        
            amp=True,             
            single_cls=True,
            
            # --- Strategie: Lücken füllen, nicht zerstören ---
            
            # 1. Geometrie (Die Lückenfüller)
            degrees=45.0,       # Rotation ist OK! QR Codes liegen oft krumm.
            translate=0.1,      # Verschieben ist sicher.
            scale=0.5,          # WICHTIG: Simuliert Distanz (Zoom). Sicher.
            
            # VORSICHTIG ANGEPASST:
            shear=1.0,          # Reduziert von 2.5 auf 1.0 (Quadrate bleiben Quadrate)
            perspective=0.0005, # Reduziert von 0.001 (Nur leichte 3D-Kippung)
            
            # 2. Struktur (Gefahrenzone)
            mosaic=0.5,         # Wahrscheinlichkeit auf 50% gesenkt. 
                                # Das heißt: Jedes 2. Bild ist ein "ganzes" Bild.
            mixup=0.0,          # AUSGESCHALTET. QR Codes sollen nicht transparent sein.
            copy_paste=0.0,     # Aus.
            
            # Am Ende wieder Mosaic aus, für Fokus auf Full-Size Bilder
            close_mosaic=10,    

            # 3. Licht & Farbe (Simuliert schlechte Kameras)
            hsv_h=0.015,        # Farbe nur minimal ändern
            hsv_s=0.5,          # Sättigung (Graue vs bunte Umgebungen)
            hsv_v=0.4,          # Helligkeit (Dunkle Lagerhalle vs Büro)
        )

    print(f"\nTraining fertig.")
    # Validierung
    model.val(imgsz=IMG_SIZE)

if __name__ == '__main__':
    run_training()