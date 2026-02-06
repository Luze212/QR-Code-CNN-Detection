from ultralytics import YOLO
import os

# ================= KONFIGURATION =================
DATA_YAML = 'dataset_final_yolov8/data.yaml'
PROJECT_NAME = 'QR_Code_Training' 
EPOCHS = 50
IMG_SIZE = 640

BATCH_SIZE = 16 
WORKERS = 1     

MODELS_TO_TRAIN = [
    # "yolov8n.pt", 
    "yolov8m.pt"]
# =================================================

def run_training():
    if not os.path.exists(DATA_YAML):
        print(f"FEHLER: '{DATA_YAML}' nicht gefunden!")
        return

    print(f"Starte High-Performance Training (Batch {BATCH_SIZE})...")

    for model_name in MODELS_TO_TRAIN:
        print(f"\n>>> TRAINIERE: {model_name} <<<")
        
        model = YOLO(model_name)
        
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            workers=WORKERS,      # Fix für Datendurchsatz auf M-Chips
            project=PROJECT_NAME, 
            name=f"train_{model_name.split('.')[0]}",
            plots=True,
            exist_ok=True,        # Überschreibt alte Ordner gleichen Namens
            device='mps',         # Apple Silicon GPU
            amp=True              # Mixed Precision (Standard für Speed)
        )
        
        print(f"Validiere {model_name}...")
        model.val()

if __name__ == '__main__':
    run_training()