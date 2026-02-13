import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. KONFIGURATION
# ==========================================

# Pfad zu deinem Modell
MODEL_ORDNER = "models/VGG16/"
MODEL_NAME = "final_Model_VGG16.keras"

# Pfad zu den Validierungsdaten 
# WICHTIG: Wir gehen direkt in den 'val' Ordner!
TEST_DATEN_PFAD = "dataset_final_boxes/val"

# Ausgabe
ERGEBNIS_ORDNER = "logs/F1-Precision_Recall"
CSV_DATEINAME = "modell_VGG16-final.csv"

# Bildgröße (muss exakt zum Training passen!)
# Für dein MobileNet/VGG war es oft 224, für das eigene CNN oft 256.
IMG_HEIGHT = 224
IMG_WIDTH = 224

# ==========================================
# 2. MODELL LADEN
# ==========================================

model_path = os.path.join(MODEL_ORDNER, MODEL_NAME)

if not os.path.exists(model_path):
    print(f"FEHLER: Das Modell wurde unter '{model_path}' nicht gefunden!")
    exit()

print(f"Lade Modell von: {model_path} ...")
model = load_model(model_path)
print("Modell erfolgreich geladen.")

# ==========================================
# 3. TESTDATEN VORBEREITEN
# ==========================================

# Nur Rescaling, keine Augmentation für den Test!
test_datagen = ImageDataGenerator(rescale=1./255)

print(f"📂 Lade Testdaten aus: {TEST_DATEN_PFAD}")

# WICHTIG: class_mode='binary' für Sigmoid-Modelle
test_generator = test_datagen.flow_from_directory(
    TEST_DATEN_PFAD,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='binary', 
    shuffle=False  # Wichtig: False, damit Reihenfolge zu den Labels passt!
)

# ==========================================
# 4. VORHERSAGE
# ==========================================

print("🔮 Führe Vorhersage durch...")
predictions = model.predict(test_generator, verbose=1)

# Binäre Entscheidung treffen (Alles über 50% ist Klasse 1)
y_pred = (predictions > 0.5).astype(int).flatten()

# Die echten Labels laden
y_true = test_generator.classes

# Klassennamen holen (z.B. ['No_QR', 'QR'])
class_labels = list(test_generator.class_indices.keys())

# ==========================================
# 5. METRIKEN & SPEICHERN
# ==========================================

# Report erstellen
report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
report_text = classification_report(y_true, y_pred, target_names=class_labels)

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(report_text)
print("="*50)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Speichern als CSV
if not os.path.exists(ERGEBNIS_ORDNER):
    os.makedirs(ERGEBNIS_ORDNER)

df_results = pd.DataFrame(report_dict).transpose()
csv_pfad = os.path.join(ERGEBNIS_ORDNER, CSV_DATEINAME)
df_results.to_csv(csv_pfad)

print(f"\nErgebnisse gespeichert unter: {csv_pfad}")

# Optional: Confusion Matrix Plot speichern
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(ERGEBNIS_ORDNER, "confusion_matrix.png"))
print("Confusion Matrix Bild gespeichert.")