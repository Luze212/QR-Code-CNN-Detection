import sys
import json
import cv2
from pathlib import Path

def main():
    project_dir = Path(__file__).resolve().parent
    default_model = project_dir / "models_yolo" / "best.pt"

    try:
        from ultralytics import YOLO
    except Exception as e:
        # immer JSON
        print(json.dumps({"boxes": [], "error": f"ultralytics import failed: {e}"}))
        return

    if len(sys.argv) < 2:
        print(json.dumps({"boxes": [], "error": "missing img_path"}))
        return

    img_path = sys.argv[1]
    # Modellpfad: 2. CLI-Argument oder Default models_yolo/best.pt
    model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_model

    # relative Pfade relativ zum Projektordner auflösen
    if not model_path.is_absolute():
        model_path = (project_dir / model_path).resolve()

    if not model_path.exists():
        print(json.dumps({"boxes": [], "error": f"model not found: {model_path}"}))
        return

    frame = cv2.imread(img_path)
    if frame is None:
        print(json.dumps({"boxes": [], "error": f"cv2.imread failed for {img_path}"}))
        return

    try:
        model = YOLO(str(model_path))
        res = model.predict(frame, conf=0.25, iou=0.5, verbose=False)[0]
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            boxes.append([x1, y1, x2, y2])
        print(json.dumps({"boxes": boxes}))
    except Exception as e:
        print(json.dumps({"boxes": [], "error": f"inference failed: {e}"}))

if __name__ == "__main__":
    main()
