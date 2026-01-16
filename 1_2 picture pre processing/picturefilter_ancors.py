import os
import shutil
from PIL import Image
import imagehash
from tqdm import tqdm

def filter_with_anchor(input_folder, output_folder, threshold=15, jump_threshold=35):
    """
    threshold: Mindestunterschied, um ein Bild zu behalten.
    jump_threshold: Ab welchem Unterschied Bild ignoriert wird
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Bilder sortiert einlesen
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    last_kept_hash = None
    kept_count = 0
    
    print(f"Analysiere {len(files)} Bilder im Anker-Verfahren...")

    for filename in tqdm(files):
        path = os.path.join(input_folder, filename)
        
        try:
            with Image.open(path) as img:
                current_hash = imagehash.dhash(img)

                # Erstes Bild immer behalten
                if last_kept_hash is None:
                    shutil.copy(path, os.path.join(output_folder, filename))
                    last_kept_hash = current_hash
                    kept_count += 1
                    continue

                # Distanz letzten gehaltenen Bild
                distance = current_hash - last_kept_hash

                # Entscheidung basierend auf Distanz
                if distance >= jump_threshold or distance >= threshold:
                    shutil.copy(path, os.path.join(output_folder, filename))
                    last_kept_hash = current_hash
                    kept_count += 1
                else:
                    # Bild Anker zu ähnlich
                    continue
                    
        except Exception as e:
            print(f"Fehler bei {filename}: {e}")

    print("-" * 30)
    print(f"Abgeschlossen! Von {len(files)} wurden {kept_count} Bilder behalten.")

if __name__ == "__main__":
    # Pfade
    QUELL_ORDNER = "/Users/lukas/Desktop/Bilder Quelle"
    ZIEL_ORDNER = "/Users/lukas/Desktop/Bilder Ziel"
    
    filter_with_anchor(QUELL_ORDNER, ZIEL_ORDNER, threshold=35, jump_threshold=55)