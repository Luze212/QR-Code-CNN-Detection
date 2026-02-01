import os
import numpy as np
import pandas as pd  # Neu: Für die CSV-Erstellung
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. KONFIGURATION (HIER ANPASSEN)
# ==========================================

# A) Eingabe: Wo liegt dein Modell und die Testdaten?
MODEL_ORDNER    = "Pfad/zu/deinem/Modellordner"    # z.B. "./models"
MODEL_NAME      = "mein_fertiges_modell.h5"        # Dateiname
TEST_DATEN_PFAD = "Pfad/zu/den/Testdaten"          # Unterordner pro Klasse beachten!

# B) Ausgabe: Wo soll die CSV-Datei hin?
ERGEBNIS_ORDNER = "Pfad/zu/den/Ergebnissen"        # z.B. "./evaluation"
CSV_DATEINAME   = "modell_evaluation.csv"          # Name der CSV-Datei

# C) Bildgröße (MUSS exakt so sein wie beim Training!)
IMG_HEIGHT = 256    # ftl: 224
IMG_WIDTH  = 256    # ftl: 224

# ==========================================
# 2. MODELL LADEN
# ==========================================

model_path = os.path.join(MODEL_ORDNER, MODEL_NAME)

if not os.path.exists(model_path):
    print(f"FEHLER: Das Modell wurde unter '{model_path}' nicht gefunden!")
else:
    print(f"Lade Modell von: {model_path} ...")
    model = load_model(model_path)
    print("Modell erfolgreich geladen.")

    # ==========================================
    # 3. TESTDATEN VORBEREITEN
    # ==========================================
    
    # Nur Rescaling, keine Augmentation für den Test!
    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Lade Testdaten...")
    test_generator = test_datagen.flow_from_directory(
        TEST_DATEN_PFAD,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # WICHTIG: Nicht mischen!
    )

    # ==========================================
    # 4. VORHERSAGE
    # ==========================================

    print("Führe Vorhersage durch...")
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # ==========================================
    # 5. METRIKEN BERECHNEN & CSV SPEICHERN
    # ==========================================

    # Report als Dictionary holen (für die Weiterverarbeitung)
    report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    
    # Report als Text für die Konsole
    report_text = classification_report(y_true, y_pred, target_names=class_labels)

    print("\n------------------------------------------------------")
    print("CLASSIFICATION REPORT (Konsole)")
    print("------------------------------------------------------")
    print(report_text)

    # In DataFrame umwandeln
    df_results = pd.DataFrame(report_dict).transpose()

    # Speicherordner erstellen, falls nicht vorhanden
    if not os.path.exists(ERGEBNIS_ORDNER):
        os.makedirs(ERGEBNIS_ORDNER)
        print(f"Ordner erstellt: {ERGEBNIS_ORDNER}")

    # Als CSV speichern
    csv_pfad = os.path.join(ERGEBNIS_ORDNER, CSV_DATEINAME)
    df_results.to_csv(csv_pfad)
    print(f"Ergebnisse erfolgreich gespeichert unter: {csv_pfad}")
