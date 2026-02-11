import cv2
import numpy as np
import qrcode
import os
import random
from glob import glob

# --- KONFIGURATION ---
BACKGROUND_DIR = "dataset_final_yolov8/train/images" # Deine echten Bilder als Hintergrund nutzen
OUTPUT_DIR = "dataset_synthetic"
NUM_IMAGES_TO_GENERATE = 500  # Wie viele neue Bilder?

def create_random_qr():
    qr = qrcode.QRCode(box_size=10, border=1)
    # Zufälliger Inhalt, damit das Muster immer anders ist!
    data = f"SYNTHETIC-{random.randint(0, 99999999)}"
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return np.array(img.convert('RGB'))

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss * 20
    return np.clip(noisy, 0, 255).astype(np.uint8)

def run_generator():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    images_dir = os.path.join(OUTPUT_DIR, "images")
    labels_dir = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Hintergründe laden (Bilder ohne QR-Code wären am besten, aber wir nehmen einfach deine Trainingsbilder)
    bg_files = glob(os.path.join(BACKGROUND_DIR, "*.jpg")) + glob(os.path.join(BACKGROUND_DIR, "*.png"))
    
    print(f"Generiere {NUM_IMAGES_TO_GENERATE} synthetische Bilder...")

    for i in range(NUM_IMAGES_TO_GENERATE):
        # 1. Hintergrund wählen
        bg_path = random.choice(bg_files)
        bg = cv2.imread(bg_path)
        if bg is None: continue
        h_bg, w_bg = bg.shape[:2]

        # 2. QR Code generieren
        qr_img = create_random_qr()
        
        # 3. QR Code zufällig skalieren (zwischen 10% und 40% der Bildgröße)
        scale = random.uniform(0.1, 0.4)
        h_qr = int(h_bg * scale)
        w_qr = int(h_qr) # Quadratisch bleiben
        qr_resized = cv2.resize(qr_img, (w_qr, h_qr), interpolation=cv2.INTER_NEAREST)

        # 4. QR Code rotieren (Augmentation)
        angle = random.randint(-45, 45)
        M = cv2.getRotationMatrix2D((w_qr//2, h_qr//2), angle, 1.0)
        qr_rotated = cv2.warpAffine(qr_resized, M, (w_qr, h_qr), borderValue=(255,255,255))

        # 5. Position wählen
        x_pos = random.randint(0, w_bg - w_qr)
        y_pos = random.randint(0, h_bg - h_qr)

        # 6. Einfügen (Simple Overlay - man könnte auch Poisson Blending nehmen)
        roi = bg[y_pos:y_pos+h_qr, x_pos:x_pos+w_qr]
        
        # Maske erstellen (schwarze Ränder vom Rotieren entfernen)
        gray_qr = cv2.cvtColor(qr_rotated, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_qr, 254, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)

        bg_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        qr_fg = cv2.bitwise_and(qr_rotated, qr_rotated, mask=mask)
        dst = cv2.add(bg_bg, qr_fg)
        
        bg[y_pos:y_pos+h_qr, x_pos:x_pos+w_qr] = dst

        # 7. Noise / Blur hinzufügen (damit es echter aussieht)
        if random.random() > 0.5:
            bg = cv2.GaussianBlur(bg, (5, 5), 0)
        
        # 8. Speichern
        filename = f"syn_{i:04d}"
        cv2.imwrite(os.path.join(images_dir, f"{filename}.jpg"), bg)

        # 9. Label schreiben (YOLO Format: class x_center y_center w h)
        # Achtung: Koordinaten müssen normalisiert (0-1) sein
        x_center = (x_pos + w_qr/2) / w_bg
        y_center = (y_pos + h_qr/2) / h_bg
        w_norm = w_qr / w_bg
        h_norm = h_qr / h_bg

        with open(os.path.join(labels_dir, f"{filename}.txt"), "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    print("✅ Fertig! Daten liegen in 'dataset_synthetic'.")
    print("Füge diese nun zu deiner data.yaml hinzu oder mische sie in den Train-Ordner.")

if __name__ == "__main__":
    run_generator()