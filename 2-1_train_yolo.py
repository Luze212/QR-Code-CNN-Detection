import os
import shutil
import sys
from pathlib import Path


def ensure_fixed_data_yaml(dataset_dir: Path) -> Path:
    """
    Roboflow export uses ../train/images etc. We create a stable YAML that works when run from project root.
    We do NOT delete the old file; we create/overwrite a fixed one named data_fixed.yaml.
    """
    fixed_yaml = dataset_dir / "data_fixed.yaml"

    content = """path: dataset_final_yolov8
train: train/images
val: valid/images
test: test/images

names:
  0: QR-Code
"""
    fixed_yaml.write_text(content, encoding="utf-8")
    return fixed_yaml


def main():
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "dataset_final_yolov8"

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset folder not found: {dataset_dir}")
        sys.exit(1)

    # 1) Create robust YAML (so we don't rely on ../ paths)
    data_yaml = ensure_fixed_data_yaml(dataset_dir)
    print(f"[OK] Using dataset config: {data_yaml}")

    # 2) Import ultralytics only after checks
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[ERROR] Could not import ultralytics. Install first:")
        print("        pip install -U ultralytics opencv-python")
        print(f"Details: {e}")
        sys.exit(1)

    # 3) Train settings (CPU-safe overnight defaults)
    # You can tweak epochs/imgsz/batch here.
    model_base = "yolov8n.pt"   # pretrained base
    imgsz = 640
    epochs = 30
    batch = 2
    workers = 0
    patience = 20

    print("[INFO] Starting training with:")
    print(f"       model={model_base}, imgsz={imgsz}, epochs={epochs}, batch={batch}, workers={workers}, patience={patience}")
    print("       (If you run out of RAM, reduce batch to 2.)")

    # 4) Start training
    model = YOLO(model_base)
    results = model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        patience=patience,
        # name="qr_yolo_train",  # optional: fixed run name
        # project="runs/detect", # optional: keep default
    )

    # 5) Locate best.pt and copy to models_yolo/best.pt
    # Ultralytics typically writes: runs/detect/train/weights/best.pt
    runs_dir = project_root / "runs" / "detect"
    best_candidates = list(runs_dir.glob("**/weights/best.pt"))

    if not best_candidates:
        print("[ERROR] Could not find best.pt under runs/detect/**/weights/best.pt")
        print("       Training may have failed or output path differs.")
        sys.exit(1)

    # Choose the newest best.pt by modification time
    best_pt = max(best_candidates, key=lambda p: p.stat().st_mtime)
    print(f"[OK] Found best.pt: {best_pt}")

    target_dir = project_root / "models_yolo"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "best.pt"

    shutil.copy2(best_pt, target_path)
    print(f"[OK] Copied to: {target_path}")

    print("\nDone. In your GUI/subprocess call use model path:")
    print(f"  {target_path}")


if __name__ == "__main__":
    main()
