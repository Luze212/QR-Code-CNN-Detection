import cv2
import os
import shutil
from skimage.metrics import structural_similarity as ssim

def filter_by_structural_similarity(input_folder, output_folder, threshold=0.80):
    """
    Diese Funktion vergleicht Bilder und kopiert nur die 'einzigartigen' 
    in einen neuen Ordner.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Liste der Bilder sortieren
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    last_image_gray = None
    kept_count = 0

    for filename in image_files:
        path = os.path.join(input_folder, filename)
        img = cv2.imread(path)
        if img is None: continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if last_image_gray is None or gray.shape != last_image_gray.shape:
            shutil.copy(path, os.path.join(output_folder, filename))
            last_image_gray = gray
            kept_count += 1
            continue

        # Ähnlichkeit berechnen
        score, _ = ssim(gray, last_image_gray, full=True)

        if score < threshold:
            shutil.copy(path, os.path.join(output_folder, filename))
            last_image_gray = gray
            kept_count += 1

    print(f"Abgeschlossen! {kept_count} Bilder wurden nach '{output_folder}' kopiert.")

#Aufruf
if __name__ == "__main__":
    # Pfade
    QUELL_ORDNER = "/Users/lukas/Desktop/Bilder Quelle"
    ZIEL_ORDNER = "/Users/lukas/Desktop/Bilder Ziel"
    
    filter_by_structural_similarity(QUELL_ORDNER, ZIEL_ORDNER, threshold=0.95)