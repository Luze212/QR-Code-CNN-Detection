import os
import shutil
from PIL import Image
import imagehash
from tqdm import tqdm

def filter_by_hashing(input_folder, output_folder, max_distance=5):
    """
    Nutzt dHash, um Bilder zu filtern. 
    max_distance: Je kleiner (z.B. 2), desto mehr Bilder werden behalten.
    Je größer (z.B. 10), desto aggressiver wird aussortiert.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    last_hash = None
    kept_count = 0

    print(f"Analysiere {len(files)} Bilder mit dHash...")

    for filename in tqdm(files):
        path = os.path.join(input_folder, filename)
        
        try:
            with Image.open(path) as img:
                # Berechne Hash
                current_hash = imagehash.dhash(img)

                if last_hash is None:
                    shutil.copy(path, os.path.join(output_folder, filename))
                    last_hash = current_hash
                    kept_count += 1
                    continue

                # Berechne differenz Hashs
                distance = current_hash - last_hash

                # Wenn größer max_distance -> Bild kopieren
                if distance > max_distance:
                    shutil.copy(path, os.path.join(output_folder, filename))
                    last_hash = current_hash
                    kept_count += 1
        except Exception as e:
            print(f"Fehler bei {filename}: {e}")

    print(f"Fertig! Behalten: {kept_count} von {len(files)}")

if __name__ == "__main__":
    # Pfade
    QUELL_ORDNER = "/Users/lukas/Desktop/Bilder Quelle"
    ZIEL_ORDNER = "/Users/lukas/Desktop/Bilder Ziel"

    filter_by_hashing(QUELL_ORDNER, ZIEL_ORDNER, max_distance=20)