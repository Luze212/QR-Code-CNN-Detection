from ultralytics import YOLO
import os
import yaml

# ================= KONFIGURATION =================
# Pfad zum Dataset-Ordner (relativ zum Skript oder absolut)
DATASET_DIR = os.path.abspath('dataset_final_yolov8')
DATA_YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

PROJECT_NAME = 'QR_Code_Training'
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 1  # Wichtig für Apple Silicon (M1/M2/M3) um Abstürze zu vermeiden

# Modell-Wahl: 'yolov8n.pt' (schnell) oder 'yolov8m.pt' (genauer)
MODEL_NAME = "yolov8m.pt" 
# =================================================

def check_and_fix_yaml():
    """
    Überprüft die data.yaml und korrigiert die Pfade auf absolute Pfade,
    damit YOLO die Bilder garantiert findet.
    """
    if not os.path.exists(DATA_YAML_PATH):
        print(f"FEHLER: '{DATA_YAML_PATH}' nicht gefunden! Bitte Pfad prüfen.")
        return False

    print(f"Prüfe Konfiguration in: {DATA_YAML_PATH}")
    
    with open(DATA_YAML_PATH, 'r') as f:
        data = yaml.safe_load(f)

    # Korrektur der Pfade: Roboflow nutzt oft '../train', wir setzen absolute Pfade
    # Annahme: Die Ordner 'train', 'valid', 'test' liegen direkt im DATASET_DIR
    data['train'] = os.path.join(DATASET_DIR, 'train', 'images')
    data['val']   = os.path.join(DATASET_DIR, 'valid', 'images')
    data['test']  = os.path.join(DATASET_DIR, 'test', 'images')
    
    # Sicherstellen, dass die Klassennamen stimmen
    data['nc'] = 1
    data['names'] = ['QR-Code']

    # Speichern der korrigierten YAML
    with open(DATA_YAML_PATH, 'w') as f:
        yaml.dump(data, f)
        
    print(" -> Pfade in data.yaml wurden auf absolute Pfade korrigiert.")
    print(f" -> Klasse gesetzt auf: {data['names']}")
    return True

def run_training():
    # 1. YAML fixen vor dem Training
    if not check_and_fix_yaml():
        return

    print(f"\nStarte Training mit {MODEL_NAME} auf Apple Silicon (MPS)...")

    # 2. Modell laden (Pretrained COCO weights)
    model = YOLO(MODEL_NAME)

    # 3. Training starten
    # YOLO erkennt automatisch 'nc: 1' aus der YAML und tauscht den Head aus.
    results = model.train(
            data=DATA_YAML_PATH,
            epochs=30,            # Weniger gegen Overfitting
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            workers=WORKERS,      
            project=PROJECT_NAME,
            name=f"train_ROBUST_{MODEL_NAME.split('.')[0]}", # Neuer Name
            plots=True,
            exist_ok=True,
            device='mps',        
            amp=True,             
            single_cls=True,
            
            # --- NEU: Gegen Overfitting (Augmentation) ---
            degrees=15.0,      
            translate=0.1,     
            scale=0.5,         
            mosaic=1.0,        
            fliplr=0.0,       
            erasing=0.4   
        )

    print(f"\nTraining abgeschlossen. Validiere {MODEL_NAME}...")
    
    # 4. Validierung
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

if __name__ == '__main__':
    run_training()