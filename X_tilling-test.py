import cv2
import numpy as np
from ultralytics import YOLO

# ================= KONFIGURATION =================
# Pfad zu deinem Modell (das, was gerade trainiert wird!)
MODEL_PATH = "models_yolo/best.pt"
# Pfad zu einem Testbild, das morgen gezeigt werden soll
IMAGE_PATH = "/Users/lukas/Desktop/Testbilder/IMG_6922.jpg" 
# =================================================

def detect_with_tiling(model, img, conf=0.25):
    """
    Versucht erst das ganze Bild. Wenn nichts gefunden wird,
    teilt es das Bild in 4 Kacheln (Zoom) und sucht dort.
    """
    results_list = []
    
    # 1. Versuch: Ganzes Bild @ 1280px
    print("Suche im Gesamtbild...")
    results = model.predict(img, imgsz=1280, conf=conf, verbose=False)
    if len(results[0].boxes) > 0:
        return results[0].plot() # Gefunden!
    
    # 2. Versuch: Tiling (Bild vierteln)
    print("Nichts gefunden. Starte Smart-Zoom (Tiling)...")
    h, w, _ = img.shape
    half_h, half_w = h // 2, w // 2
    
    # Die 4 Quadranten definieren
    crops = [
        (0, 0, half_w, half_h),       # Oben Links
        (half_w, 0, w, half_h),       # Oben Rechts
        (0, half_h, half_w, h),       # Unten Links
        (half_w, half_h, w, h)        # Unten Rechts
    ]
    
    found_something = False
    
    for x1, y1, x2, y2 in crops:
        crop = img[y1:y2, x1:x2]
        # Crop analysieren
        crop_results = model.predict(crop, imgsz=1280, conf=conf, verbose=False)
        
        if len(crop_results[0].boxes) > 0:
            found_something = True
            # Zeichne die Boxen in den Crop
            annotated_crop = crop_results[0].plot()
            # Setze den annotierten Crop zurück ins Originalbild
            img[y1:y2, x1:x2] = annotated_crop
            
    if found_something:
        print(" -> QR-Code im Zoom gefunden!")
    else:
        print(" -> Auch im Zoom nichts gefunden.")
        
    return img

if __name__ == "__main__":
    # Modell laden
    try:
        model = YOLO(MODEL_PATH)
        
        # Bild laden
        original_img = cv2.imread(IMAGE_PATH)
        if original_img is None:
            print(f"Fehler: Bild {IMAGE_PATH} nicht gefunden.")
            exit()

        # Magische Erkennung
        final_image = detect_with_tiling(model, original_img)
        
        # Anzeigen (Drücke 'q' zum Beenden)
        cv2.imshow("Smart Detection (Press Q to quit)", final_image)
        # Optional: Speichern für die PowerPoint
        cv2.imwrite("ergebnis_demo.jpg", final_image) 
        
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Modell noch nicht bereit oder Pfad falsch: {e}")