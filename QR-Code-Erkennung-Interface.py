import sys
import os
import glob
import numpy as np
import PySide6

# --- PySide6 IMPORTS ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QComboBox, QLineEdit, QFileDialog, 
                             QSlider, QScrollArea, QFrame, QProgressBar, QSizePolicy,
                             QListWidget, QListWidgetItem, QTextEdit)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer, QLocale
from PySide6.QtGui import QPixmap, QImage, QDoubleValidator, QIcon, QKeyEvent, QPainter, QPen, QColor, QFont

# --- MATPLOTLIB FÜR WINKEL-PLOT ---
import matplotlib
matplotlib.use('QtAgg') 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

# --- KONFIGURATION ---
MODEL_DIRS = ["models", "models_tfl", "models_cnn"]
IMG_SIZE = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --- STYLESHEET ---
STYLESHEET = """
QMainWindow { background-color: #2b2b2b; }
QWidget { color: #ffffff; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }

/* Sidebar Styling */
QScrollArea { border: none; background-color: #1e1e1e; }
QWidget#SidebarContent { background-color: #1e1e1e; }

/* GEOMETRIE */
QComboBox, QLineEdit, QPushButton { height: 30px; }

/* Inputs & Buttons */
QComboBox, QLineEdit, QTextEdit, QListWidget {
    background-color: #333; border: 1px solid #555; border-radius: 4px;
    padding-left: 5px; color: white; selection-background-color: #1f6aa5;
}

QComboBox::drop-down {
    border: none; background: #444; width: 20px;
    border-top-right-radius: 4px; border-bottom-right-radius: 4px;
}
QComboBox QAbstractItemView {
    background-color: #333; color: white; selection-background-color: #1f6aa5;
    outline: none; border: 1px solid #555;
}

QPushButton {
    background-color: #3a3a3a; border: 1px solid #555; border-radius: 4px;
    color: white; font-weight: bold;
}
QPushButton:hover { background-color: #4a4a4a; border-color: #666; }
QPushButton:pressed { background-color: #222; }
QPushButton:disabled { background-color: #2a2a2a; color: #555; border-color: #333; }

/* Action Buttons */
QPushButton#RunButton {
    background-color: #1f6aa5; border: none; font-size: 15px; height: 40px;
}
QPushButton#RunButton:hover { background-color: #2a7bb6; }

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #333; height: 6px; background: #222; margin: 2px 0; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #1f6aa5; border: 1px solid #1f6aa5; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}

/* Labels */
QLabel#ResultLabel {
    background-color: #222; border: 2px solid #444; border-radius: 6px;
    padding: 10px; font-weight: bold; font-size: 16px; color: #888;
}
QLabel#ResultLabel[status="success"] { border-color: #2ea043; color: #2ea043; background-color: #1a2e1f; }
QLabel#ResultLabel[status="fail"]    { border-color: #da3633; color: #da3633; background-color: #2e1a1a; }

QListWidget::item { padding: 5px; }
QListWidget::item:selected { background-color: #1f6aa5; color: white; }

/* Image Area */
QLabel#ImageDisplay {
    background-color: #181818; border: 2px dashed #333; border-radius: 8px; padding: 6px; 
}
QLabel#ImageDisplay[status="success"] { border: 4px solid #2ea043; background-color: #122215; }
QLabel#ImageDisplay[status="fail"] { border: 4px solid #da3633; background-color: #2e1a1a; }
QLabel#ImageDisplay[status="neutral"] { border: 2px dashed #333; background-color: #181818; }

/* Plot Container (Rahmen für den Plot-Bereich) */
QWidget#PlotContainer {
    border: 2px dashed #444;
    border-radius: 6px;
    background-color: #1e1e1e;
}

QFrame[frameShape="4"] { color: #444; margin-top: 15px; margin-bottom: 15px; }
"""

# =============================================================================
# --- WORKER THREADS ---
# =============================================================================

# 1. CNN WORKER
class PredictionWorker(QThread):
    progress = Signal(int)
    finished = Signal(dict) 
    error = Signal(str)

    def __init__(self, model_path, image_paths):
        super().__init__()
        self.model_path = model_path
        self.image_paths = image_paths

    def run(self):
        try:
            model = tf.keras.models.load_model(self.model_path)
            results = {}
            total = len(self.image_paths)
            for i, path in enumerate(self.image_paths):
                try:
                    img = load_img(path, target_size=IMG_SIZE)
                    x = img_to_array(img) / 255.0
                    x = np.expand_dims(x, axis=0)
                    score = model.predict(x, verbose=0)[0][0]
                    results[path] = score
                except Exception: pass
                self.progress.emit(int(((i + 1) / total) * 100))
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

# 2. YOLO WORKER
class YoloWorker(QThread):
    progress = Signal(int)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        # ==============================================================================
        # --- SCHNITTSTELLE / INTERFACE BESCHREIBUNG FÜR BACKEND-ENTWICKLER ---
        # ==============================================================================
        # ZIEL:
        # Dieses Modul soll auf dem Bild (self.image_path) QR-Codes erkennen und deren
        # Position als Bounding Boxen zurückgeben.
        #
        # INPUT:
        # self.image_path (str): Absoluter Pfad zur Bilddatei.
        #
        # AUFGABEN:
        # 1. Bild laden.
        # 2. Objekterkennungsmodell (z.B. YOLOv8) laden und anwenden.
        # 3. Filtern nach Klasse 'QR-Code' (falls Modell mehrere Klassen kann).
        #
        # OUTPUT (Rückgabeformat):
        # Eine Liste von Dictionaries. Jedes Dictionary repräsentiert einen erkannten QR-Code.
        # Format:
        # [
        #   {
        #     'id': 1,           # Eindeutige ID für diesen Durchlauf (int)
        #     'x': 100,          # X-Koordinate obere linke Ecke (Pixel)
        #     'y': 50,           # Y-Koordinate obere linke Ecke (Pixel)
        #     'w': 200,          # Breite der Box (Pixel)
        #     'h': 200,          # Höhe der Box (Pixel)
        #     'confidence': 0.95 # Wahrscheinlichkeit (0.0 bis 1.0)
        #   },
        #   ... weitere Boxen ...
        # ]
        # ==============================================================================
        try:
            self.progress.emit(10)
            
            # --- TODO: HIER CODE EINFÜGEN ---
            # Beispiel:
            # model = YOLO("best.pt")
            # results = model.predict(self.image_path)
            # boxes = parse_results(results)
            
            # Simuliere Ladezeit
            self.msleep(800) 
            self.progress.emit(50)

            # --- DUMMY DATEN (Zum Testen der GUI) ---
            dummy_boxes = [
                {'id': 1, 'x': 50, 'y': 50, 'w': 100, 'h': 100, 'confidence': 0.92},
                {'id': 2, 'x': 200, 'y': 200, 'w': 80, 'h': 80, 'confidence': 0.65},
                {'id': 3, 'x': 350, 'y': 100, 'w': 120, 'h': 120, 'confidence': 0.45}
            ]
            
            self.progress.emit(100)
            self.finished.emit(dummy_boxes)
        except Exception as e:
            self.error.emit(str(e))

# 3. READER WORKER
class ReaderWorker(QThread):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, image_path, boxes):
        super().__init__()
        self.image_path = image_path
        self.boxes = boxes

    def run(self):
        # ==============================================================================
        # --- SCHNITTSTELLE / INTERFACE BESCHREIBUNG FÜR BACKEND-ENTWICKLER ---
        # ==============================================================================
        # ZIEL:
        # Für jede gefundene Box (aus YOLO) soll der Inhalt des QR-Codes ausgelesen werden.
        # Es sollen mehrere Bibliotheken (Reader) getestet werden.
        #
        # INPUT:
        # self.image_path (str): Pfad zum Originalbild.
        # self.boxes (list): Die Liste der Boxen, die der YoloWorker zurückgegeben hat.
        #
        # AUFGABEN:
        # 1. Originalbild laden.
        # 2. Für jede Box in self.boxes:
        #    a. Den Bereich (ROI) basierend auf x,y,w,h ausschneiden.
        #    b. Diesen Ausschnitt an verschiedene Reader übergeben (z.B. Pyzbar, OpenCV, ZXing).
        #
        # OUTPUT (Rückgabeformat):
        # Eine flache Liste von Ergebnissen pro Reader und pro Code.
        # Format:
        # [
        #   {'reader': 'Pyzbar', 'code_id': 1, 'success': True, 'content': 'http://google.com'},
        #   {'reader': 'OpenCV', 'code_id': 1, 'success': False, 'content': ''},
        #   {'reader': 'Pyzbar', 'code_id': 2, 'success': True, 'content': 'Text-123'},
        #   ...
        # ]
        # ==============================================================================
        try:
            results = []
            
            # --- TODO: HIER CODE EINFÜGEN ---
            # image = cv2.imread(self.image_path)
            # for box in self.boxes:
            #     roi = image[box['y']:box['y']+box['h'], box['x']:box['x']+box['w']]
            #     text_pyzbar = read_pyzbar(roi)
            #     results.append(...)
            
            # --- DUMMY DATEN ---
            for box in self.boxes:
                results.append({'reader': 'Pyzbar', 'code_id': box['id'], 'success': True, 'content': f"Daten ID {box['id']}"})
                results.append({'reader': 'OpenCV', 'code_id': box['id'], 'success': False, 'content': ""})
                
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

# 4. ANGLE WORKER
class AngleWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, image_path, box_id, boxes):
        super().__init__()
        self.image_path = image_path
        self.target_id = box_id
        self.boxes = boxes

    def run(self):
        # ==============================================================================
        # --- SCHNITTSTELLE / INTERFACE BESCHREIBUNG FÜR BACKEND-ENTWICKLER ---
        # ==============================================================================
        # ZIEL:
        # Berechnung der räumlichen Orientierung (Winkel) des ausgewählten QR-Codes zur Kamera.
        #
        # INPUT:
        # self.image_path (str): Pfad zum Bild.
        # self.target_id (int): Die ID des QR-Codes, der analysiert werden soll (vom User gewählt).
        # self.boxes (list): Liste aller Boxen, um die Koordinaten der Ziel-ID zu finden.
        #
        # AUFGABEN:
        # 1. Box mit id == self.target_id heraussuchen.
        # 2. Bildausschnitt analysieren (z.B. Identifikation der 3 Finder-Pattern Ecken).
        # 3. PnP (Perspective-n-Point) Algorithmus oder ähnliches anwenden, um die Lage im Raum zu bestimmen.
        # 4. Normalenvektor der QR-Code-Ebene berechnen.
        #
        # OUTPUT (Rückgabeformat):
        # Ein Dictionary mit Vektoren für die 3D-Visualisierung.
        # Format:
        # {
        #   'normal_vec': [x, y, z],  # Der Vektor, der senkrecht aus dem QR-Code zeigt (Länge egal, wird normiert)
        #   'view_vec': [0, 0, 1],    # Der Vektor der Kamera (meist Z-Achse)
        #   'angle_deg': 45.5         # Der berechnete Winkel zwischen Normale und Kameraachse in Grad
        # }
        # ==============================================================================
        try:
            # --- TODO: HIER CODE EINFÜGEN ---
            
            # --- DUMMY DATEN ---
            import random
            nx = random.uniform(-1, 1)
            # Simulierter Normalenvektor
            result = {
                'normal_vec': [nx, random.uniform(-1, 1), 1.0], 
                'view_vec': [0, 0, 1], 
                'angle_deg': 42.5
            }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# =============================================================================
# --- HAUPTFENSTER ---
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KI QR-Code Erkennung")
        self.resize(1200, 900)
        self.setMinimumSize(900, 600)
        
        self.image_paths = []
        self.current_index = 0
        self.scores = {}       
        self.yolo_boxes = {}   
        self.current_yolo_boxes = [] 
        
        self.setup_ui()
        self.scan_models()
        self.setStyleSheet(STYLESHEET)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SIDEBAR ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFixedWidth(400)
        
        sidebar_content = QWidget()
        sidebar_content.setObjectName("SidebarContent")
        self.sidebar_layout = QVBoxLayout(sidebar_content)
        self.sidebar_layout.setContentsMargins(20, 20, 20, 20)
        self.sidebar_layout.setSpacing(15)

        # --- 1. CNN ---
        self.add_header("1. CNN Anwenden")
        self.combo_model = QComboBox()
        self.sidebar_layout.addWidget(self.combo_model)

        path_frame = QFrame()
        path_layout = QHBoxLayout(path_frame)
        path_layout.setContentsMargins(0,0,0,0)
        self.input_path = QLineEdit()
        self.input_path.setReadOnly(True)
        self.input_path.setPlaceholderText("Pfad...")
        btn_file = QPushButton("📄"); btn_file.setFixedWidth(40); btn_file.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_folder = QPushButton("📂"); btn_folder.setFixedWidth(40); btn_folder.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_file.clicked.connect(self.open_file_dialog)
        btn_folder.clicked.connect(self.open_folder_dialog)
        path_layout.addWidget(self.input_path); path_layout.addWidget(btn_file); path_layout.addWidget(btn_folder)
        self.sidebar_layout.addWidget(path_frame)

        # Threshold CNN
        cnn_thresh_frame = QFrame()
        cnn_thresh_layout = QHBoxLayout(cnn_thresh_frame)
        cnn_thresh_layout.setContentsMargins(0,0,0,0)
        
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(0, 100)
        self.slider_thresh.setValue(70)
        
        self.input_thresh = QLineEdit("0.70")
        self.input_thresh.setFixedWidth(60)
        self.input_thresh.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        val_cnn = QDoubleValidator(0.00, 1.00, 2)
        val_cnn.setNotation(QDoubleValidator.Notation.StandardNotation)
        val_cnn.setLocale(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
        self.input_thresh.setValidator(val_cnn)

        self.slider_thresh.valueChanged.connect(self.sync_cnn_input)
        self.input_thresh.editingFinished.connect(self.sync_cnn_slider)
        
        cnn_thresh_layout.addWidget(QLabel("Min. Konfidenz:"))
        cnn_thresh_layout.addWidget(self.slider_thresh)
        cnn_thresh_layout.addWidget(self.input_thresh)
        self.sidebar_layout.addWidget(cnn_thresh_frame)

        # Nav
        nav_frame = QFrame()
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(0,0,0,0)
        self.btn_prev = QPushButton("<"); self.btn_prev.setEnabled(False); self.btn_prev.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_next = QPushButton(">"); self.btn_next.setEnabled(False); self.btn_next.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.lbl_filename = QLabel("Keine Bilder")
        self.lbl_filename.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        nav_layout.addWidget(self.btn_prev); nav_layout.addWidget(self.lbl_filename); nav_layout.addWidget(self.btn_next)
        self.sidebar_layout.addWidget(nav_frame)

        self.btn_run = QPushButton("CNN ANWENDEN") 
        self.btn_run.setObjectName("RunButton"); self.btn_run.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_run.clicked.connect(self.start_cnn)
        self.sidebar_layout.addWidget(self.btn_run)
        
        self.prog_cnn = QProgressBar(); self.prog_cnn.setFixedHeight(5); self.prog_cnn.setVisible(False)
        self.sidebar_layout.addWidget(self.prog_cnn)
        self.lbl_result = QLabel("Bereit"); self.lbl_result.setObjectName("ResultLabel"); self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sidebar_layout.addWidget(self.lbl_result)

        self.add_separator()

        # --- 2. YOLO ---
        self.add_header("2. QR-Code Ortung")
        self.btn_yolo = QPushButton("QR-CODES ORTEN")
        self.btn_yolo.setObjectName("RunButton"); self.btn_yolo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_yolo.clicked.connect(self.start_yolo)
        self.sidebar_layout.addWidget(self.btn_yolo)

        # Threshold YOLO
        yolo_thresh_frame = QFrame()
        yolo_thresh_layout = QHBoxLayout(yolo_thresh_frame)
        yolo_thresh_layout.setContentsMargins(0,0,0,0)
        
        self.slider_yolo = QSlider(Qt.Orientation.Horizontal)
        self.slider_yolo.setRange(0, 100)
        self.slider_yolo.setValue(50)
        
        self.input_yolo_thresh = QLineEdit("0.50")
        self.input_yolo_thresh.setFixedWidth(60)
        self.input_yolo_thresh.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        val_yolo = QDoubleValidator(0.00, 1.00, 2)
        val_yolo.setNotation(QDoubleValidator.Notation.StandardNotation)
        val_yolo.setLocale(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
        self.input_yolo_thresh.setValidator(val_yolo)

        self.slider_yolo.valueChanged.connect(self.sync_yolo_input)
        self.input_yolo_thresh.editingFinished.connect(self.sync_yolo_slider)
        
        yolo_thresh_layout.addWidget(QLabel("Min. Konfidenz:"))
        yolo_thresh_layout.addWidget(self.slider_yolo)
        yolo_thresh_layout.addWidget(self.input_yolo_thresh)
        self.sidebar_layout.addWidget(yolo_thresh_frame)

        self.prog_yolo = QProgressBar(); self.prog_yolo.setFixedHeight(5); self.prog_yolo.setVisible(False)
        self.sidebar_layout.addWidget(self.prog_yolo)
        self.lbl_yolo_res = QLabel("Noch nicht geortet"); self.lbl_yolo_res.setObjectName("ResultLabel"); self.lbl_yolo_res.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sidebar_layout.addWidget(self.lbl_yolo_res)

        self.add_separator()

        # --- 3. READER ---
        self.add_header("3. QR-Inhalt Lesen")
        self.btn_read = QPushButton("INHALT LESEN")
        self.btn_read.setObjectName("RunButton"); self.btn_read.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_read.clicked.connect(self.start_reading)
        self.sidebar_layout.addWidget(self.btn_read)

        self.list_reader = QListWidget()
        self.list_reader.setFixedHeight(100)
        self.list_reader.itemClicked.connect(self.show_reader_content)
        self.sidebar_layout.addWidget(self.list_reader)

        self.txt_content = QTextEdit()
        self.txt_content.setReadOnly(True)
        self.txt_content.setPlaceholderText("Eintrag in der Liste auswählen")
        self.txt_content.setFixedHeight(80)
        self.sidebar_layout.addWidget(self.txt_content)

        self.add_separator()

        # --- 4. WINKEL ---
        self.add_header("4. Winkel Bestimmung")
        angle_sel_layout = QHBoxLayout()
        angle_sel_layout.addWidget(QLabel("QR-Code ID:"))
        self.combo_qr_id = QComboBox(); self.combo_qr_id.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        angle_sel_layout.addWidget(self.combo_qr_id)
        self.sidebar_layout.addLayout(angle_sel_layout)

        self.btn_angle = QPushButton("WINKEL BESTIMMEN")
        self.btn_angle.setObjectName("RunButton"); self.btn_angle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_angle.clicked.connect(self.start_angle)
        self.sidebar_layout.addWidget(self.btn_angle)

        # Plot Container erstellen (damit der Rahmen sichtbar ist)
        self.plot_container = QWidget()
        self.plot_container.setObjectName("PlotContainer")
        self.plot_container.setMinimumHeight(350)
        plot_layout = QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_figure = Figure(figsize=(4, 4), dpi=100)
        self.plot_figure.patch.set_facecolor('#1e1e1e')
        self.plot_canvas = FigureCanvasQTAgg(self.plot_figure)
        # Canvas zum Layout hinzufügen
        plot_layout.addWidget(self.plot_canvas)
        
        self.sidebar_layout.addWidget(self.plot_container)

        self.sidebar_layout.addStretch()
        scroll_area.setWidget(sidebar_content)
        main_layout.addWidget(scroll_area, stretch=1)

        # --- RECHTE SEITE ---
        self.image_container = QWidget()
        self.image_container.setStyleSheet("background-color: #222;")
        img_layout = QVBoxLayout(self.image_container)
        self.lbl_image = QLabel("Bitte Ordner oder Datei wählen")
        self.lbl_image.setObjectName("ImageDisplay")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setScaledContents(False) 
        img_layout.addWidget(self.lbl_image)
        main_layout.addWidget(self.image_container, stretch=2)

    # --- HELPER ---
    def add_header(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #ccc; font-weight: bold; font-size: 16px; margin-top: 10px;")
        self.sidebar_layout.addWidget(lbl)

    def add_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.sidebar_layout.addWidget(line)

    def scan_models(self):
        self.combo_model.clear()
        found = False
        for d in MODEL_DIRS:
            if os.path.exists(d):
                files = glob.glob(os.path.join(d, "*.keras")) + glob.glob(os.path.join(d, "*.h5"))
                for f in files: self.combo_model.addItem(f); found = True
        if not found: self.combo_model.addItem("Keine Modelle"); self.btn_run.setEnabled(False)

    # --- FILE DIALOGS (FIX: KÜRZEN) ---
    def open_file_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Bild wählen", "", "Bilder (*.png *.jpg *.jpeg *.bmp *.webp)")
        if f: 
            self.image_paths = [f]
            self.reset_viewer()
            # Text kürzen
            text = os.path.basename(f)
            if len(text) > 30: text = text[:15] + "..." + text[-10:]
            self.input_path.setText(text)
    
    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Ordner wählen")
        if folder:
            self.image_paths = []
            for ext in VALID_EXTENSIONS: self.image_paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
            try:
                subdirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
                for sd in subdirs:
                    for ext in VALID_EXTENSIONS: self.image_paths.extend(glob.glob(os.path.join(sd, f"*{ext}")))
            except: pass
            self.image_paths = sorted(list(set(self.image_paths)))
            if self.image_paths: 
                self.reset_viewer()
                # Text kürzen
                text = f"{os.path.basename(folder)} ({len(self.image_paths)})"
                if len(text) > 30: text = text[:15] + "..." + text[-10:]
                self.input_path.setText(text)
            else: self.lbl_filename.setText("0 Bilder")

    # --- NAVIGATION (FIX: FILENAME KÜRZEN) ---
    def reset_viewer(self):
        self.current_index = 0
        self.scores = {}
        self.yolo_boxes = {}
        self.current_yolo_boxes = []
        self.update_image_view()
        self.setFocus()

    def prev_image(self):
        if self.current_index > 0: self.current_index -= 1; self.update_image_view()
    def next_image(self):
        if self.current_index < len(self.image_paths) - 1: self.current_index += 1; self.update_image_view()

    def keyPressEvent(self, event: QKeyEvent):
        if self.input_thresh.hasFocus() or self.input_yolo_thresh.hasFocus():
            super().keyPressEvent(event); return
        if event.key() == Qt.Key.Key_Left and self.btn_prev.isEnabled(): self.prev_image()
        elif event.key() == Qt.Key.Key_Right and self.btn_next.isEnabled(): self.next_image()
        else: super().keyPressEvent(event)

    # --- DISPLAY LOGIC ---
    def update_image_view(self):
        if not self.image_paths: return
        path = self.image_paths[self.current_index]
        
        # Text kürzen für unteres Label
        fname = os.path.basename(path)
        if len(fname) > 25: fname = fname[:12] + "..." + fname[-10:]
        self.lbl_filename.setText(f"{self.current_index+1}/{len(self.image_paths)}: {fname}")
        
        self.btn_prev.setEnabled(self.current_index > 0)
        self.btn_next.setEnabled(self.current_index < len(self.image_paths) - 1)

        base_pixmap = QPixmap(path)
        if base_pixmap.isNull(): return
        avail_size = self.lbl_image.size() - QSize(12, 12)
        scaled_pixmap = base_pixmap.scaled(avail_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        if path in self.yolo_boxes:
            self.draw_overlays(scaled_pixmap, self.yolo_boxes[path], base_pixmap.size())

        self.lbl_image.setPixmap(scaled_pixmap)
        self.update_cnn_visuals()

    def draw_overlays(self, pixmap, boxes, original_size):
        painter = QPainter(pixmap)
        pen = QPen(QColor("#00ff00")); pen.setWidth(3)
        painter.setPen(pen)
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))

        scale_x = pixmap.width() / original_size.width()
        scale_y = pixmap.height() / original_size.height()
        thresh = self.slider_yolo.value() / 100.0
        visible_count = 0
        ids = []

        for box in boxes:
            if box['confidence'] >= thresh:
                visible_count += 1
                ids.append(str(box['id']))
                x = int(box['x'] * scale_x); y = int(box['y'] * scale_y)
                w = int(box['w'] * scale_x); h = int(box['h'] * scale_y)
                painter.drawRect(x, y, w, h)
                painter.fillRect(x, y - 20, 100, 20, QColor(0,0,0, 150))
                painter.drawText(x + 5, y - 5, f"#{box['id']} ({box['confidence']:.2f})")
        painter.end()
        
        self.lbl_yolo_res.setText(f"{visible_count} QR-Codes gefunden")
        self.current_yolo_boxes = boxes
        
        current_id = self.combo_qr_id.currentText()
        self.combo_qr_id.clear(); self.combo_qr_id.addItems(ids)
        if current_id in ids: self.combo_qr_id.setCurrentText(current_id)

    # --- SYNC FUNCTIONS ---
    def sync_cnn_input(self, val):
        self.input_thresh.setText(f"{val/100:.2f}")
        self.update_cnn_visuals()
    
    def sync_cnn_slider(self):
        try:
            val = float(self.input_thresh.text().replace(',', '.'))
            self.slider_thresh.setValue(int(val * 100))
            self.update_cnn_visuals()
        except ValueError: pass

    def sync_yolo_input(self, val):
        self.input_yolo_thresh.setText(f"{val/100:.2f}")
        self.update_image_view()

    def sync_yolo_slider(self):
        try:
            val = float(self.input_yolo_thresh.text().replace(',', '.'))
            self.slider_yolo.setValue(int(val * 100))
            self.update_image_view()
        except ValueError: pass

    # --- CNN START ---
    def start_cnn(self):
        model = self.combo_model.currentText()
        if not self.image_paths or "Keine" in model: return
        self.btn_run.setEnabled(False); self.btn_run.setText("LÄUFT...")
        self.prog_cnn.setVisible(True); self.prog_cnn.setValue(0)
        self.lbl_result.setText("Berechne..."); self.lbl_result.setProperty("status", "neutral"); self.refresh_style()
        self.cnn_worker = PredictionWorker(model, self.image_paths)
        self.cnn_worker.progress.connect(self.prog_cnn.setValue)
        self.cnn_worker.finished.connect(self.on_cnn_finished)
        self.cnn_worker.start()

    def on_cnn_finished(self, res):
        self.scores = res
        self.btn_run.setEnabled(True); self.btn_run.setText("CNN ANWENDEN")
        self.prog_cnn.setVisible(False)
        self.update_cnn_visuals()
        self.setFocus()

    def update_cnn_visuals(self):
        if not self.image_paths: return
        path = self.image_paths[self.current_index]
        if path in self.scores:
            sc = self.scores[path]
            is_qr = sc >= (self.slider_thresh.value()/100.0)
            txt = "QR-CODE ERKANNT" if is_qr else "KEIN QR-CODE"
            self.lbl_result.setText(f"{txt} ({sc:.2f})")
            prop = "success" if is_qr else "fail"
            self.lbl_result.setProperty("status", prop)
            self.lbl_image.setProperty("status", prop)
        else:
            self.lbl_result.setText("Bereit")
            self.lbl_result.setProperty("status", "neutral")
            self.lbl_image.setProperty("status", "neutral")
        self.refresh_style()

    # --- YOLO START ---
    def start_yolo(self):
        if not self.image_paths: return
        path = self.image_paths[self.current_index]
        self.btn_yolo.setEnabled(False); self.btn_yolo.setText("ORTUNG LÄUFT...")
        self.prog_yolo.setVisible(True); self.prog_yolo.setValue(0)
        self.yolo_worker = YoloWorker(path)
        self.yolo_worker.progress.connect(self.prog_yolo.setValue)
        self.yolo_worker.finished.connect(self.on_yolo_finished)
        self.yolo_worker.start()

    def on_yolo_finished(self, boxes):
        path = self.image_paths[self.current_index]
        self.yolo_boxes[path] = boxes
        self.btn_yolo.setEnabled(True); self.btn_yolo.setText("QR-CODES ORTEN")
        self.prog_yolo.setVisible(False)
        self.update_image_view()
        self.setFocus()

    # --- READER START ---
    def start_reading(self):
        if not self.image_paths: return
        path = self.image_paths[self.current_index]
        if path not in self.yolo_boxes or not self.yolo_boxes[path]:
            self.txt_content.setText("FEHLER: Bitte erst QR-Codes orten (YOLO)!")
            return
        self.btn_read.setEnabled(False); self.btn_read.setText("LESE...")
        self.list_reader.clear()
        self.reader_worker = ReaderWorker(path, self.yolo_boxes[path])
        self.reader_worker.finished.connect(self.on_read_finished)
        self.reader_worker.start()

    def on_read_finished(self, results):
        self.btn_read.setEnabled(True); self.btn_read.setText("INHALT LESEN")
        for res in results:
            icon = "✅" if res['success'] else "❌"
            item = QListWidgetItem(f"{icon} {res['reader']} (ID: #{res['code_id']})")
            item.setData(Qt.ItemDataRole.UserRole, res['content'])
            self.list_reader.addItem(item)
        self.setFocus()

    def show_reader_content(self, item):
        content = item.data(Qt.ItemDataRole.UserRole)
        self.txt_content.setText(content if content else "Kein Inhalt lesbar.")

    # --- ANGLE START ---
    def start_angle(self):
        if not self.image_paths: return
        path = self.image_paths[self.current_index]
        target_id_str = self.combo_qr_id.currentText()
        if not target_id_str: return
        self.angle_worker = AngleWorker(path, int(target_id_str), self.yolo_boxes.get(path, []))
        self.angle_worker.finished.connect(self.on_angle_finished)
        self.angle_worker.start()

    def on_angle_finished(self, data):
        self.plot_figure.clear()
        ax = self.plot_figure.add_subplot(111, projection='3d')
        ax.set_facecolor('#1e1e1e')
        n = data['normal_vec']
        ax.quiver(0, 0, 0, n[0], n[1], n[2], color='#1f6aa5', length=1.0, normalize=True, label='Normale')
        v = data['view_vec']
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color='white', length=1.0, normalize=True, linestyle='dashed', label='Kamera')
        xx, yy = np.meshgrid(range(-1, 2), range(-1, 2))
        z = (-n[0] * xx - n[1] * yy) / n[2]
        ax.plot_surface(xx, yy, z, alpha=0.3, color='green')
        ax.set_title(f"Winkel: {data['angle_deg']}°", color='white')
        ax.tick_params(colors='gray')
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.legend()
        self.plot_canvas.draw()
        self.setFocus()

    def refresh_style(self):
        self.lbl_result.style().unpolish(self.lbl_result); self.lbl_result.style().polish(self.lbl_result)
        self.lbl_image.style().unpolish(self.lbl_image); self.lbl_image.style().polish(self.lbl_image)

    def resizeEvent(self, event):
        if self.image_paths: QTimer.singleShot(100, self.update_image_view)
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())