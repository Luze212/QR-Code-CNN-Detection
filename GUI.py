import sys
import os

# --- WICHTIGE FIXES FÜR MACOS & VS CODE ---
# 1. Accessibility abschalten: Verhindert den Absturz/Hänger in VS Code
os.environ['QT_ACCESSIBILITY'] = '0' 

# 2. Qt Plugin Pfad finden (gegen "Cocoa plugin not found" Fehler)
import PyQt6
def find_qt_plugins():
    base_path = os.path.dirname(PyQt6.__file__)
    paths = [
        os.path.join(base_path, 'Qt6', 'plugins'),
        os.path.join(base_path, '..', 'PyQt6_Qt6', 'plugins'),
        os.path.join(base_path, 'plugins')
    ]
    for p in paths:
        if os.path.exists(os.path.join(p, 'platforms', 'libqcocoa.dylib')): return p
    return paths[0]

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = find_qt_plugins()
# ------------------------------------------

import matplotlib
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QComboBox, QLineEdit, QFileDialog, 
                             QFrame, QTextEdit, QSlider, QSizePolicy, QListView, QMenu, QProgressBar)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QDoubleValidator, QAction, QCursor, QFont

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

# --- TENSORFLOW IMPORTS ---
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore

# --- YOLO IMPORTS ---
import cv2
#Test Keine Rückmeldung
from PyQt6.QtCore import QThread, pyqtSignal
import subprocess, sys, json, os

# --- KONFIGURATION ---
MODEL_DIRS = ["models", "models_tfl"]
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
IMG_SIZE = (300, 300)

ARROW_ICON = b"iVBORw0KGgoAAAANSUhEUgAAAAwAAAAMCAYAAABWdVznAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH5gIbDBQyR5yj+AAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAQUlEQVQoz2JwYGBgYGBgOMCAB4yMjEiCsAF0PRQM6MK4NKAriG4CsiG4FKAqCEsBqoKoIrI8iGoCsiKibgJ5J7o8AHv0C/W+7R+jAAAAAElFTkSuQmCC"

# --- STYLESHEET ---
DARK_STYLESHEET = f"""
QMainWindow {{ background-color: #2b2b2b; }}
QWidget {{ color: #ffffff; font-size: 14px; }}
QFrame#Sidebar {{ background-color: #1e1e1e; border-right: 1px solid #3e3e3e; }}
QLabel#Header {{ font-size: 18px; font-weight: bold; color: #ffffff; padding: 10px; }}
QLabel {{ color: #ddd; }}

QPushButton {{
    background-color: #3a3a3a; border: 1px solid #555; border-radius: 4px; padding: 6px; color: white;
}}
QPushButton:hover {{ background-color: #4a4a4a; }}
QPushButton:pressed {{ background-color: #2a2a2a; }}

QPushButton#ActionBtn {{ background-color: #1f6aa5; border: none; }}
QPushButton#ActionBtn:hover {{ background-color: #2a7bb6; }}
QPushButton#SuccessBtn {{ background-color: #2ea043; border: none; }}

QPushButton#NavBtn {{
    background-color: #3a3a3a; border: 1px solid #555; border-radius: 4px; font-weight: bold;
}}

QComboBox {{
    background-color: #3a3a3a; border: 1px solid #555; border-radius: 4px; padding: 4px 10px;
    min-height: 28px; selection-background-color: #1f6aa5;
}}
QComboBox QLineEdit {{ background: transparent; border: none; color: white; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: top right; width: 30px;
    border-left: 1px solid #555; background-color: #444;
    border-top-right-radius: 4px; border-bottom-right-radius: 4px;
}}
QComboBox::down-arrow {{
    image: url(data:image/png;base64,{str(ARROW_ICON, 'utf-8')}); width: 12px; height: 12px;
}}
QComboBox QAbstractItemView {{
    border: 1px solid #555; background-color: #3a3a3a; color: white;
    selection-background-color: #1f6aa5; outline: none;
}}

QFrame#MergedPathFrame {{
    background-color: #3a3a3a; border: 1px solid #555; border-radius: 4px;
}}
QLineEdit#FramelessInput {{
    background: transparent; border: none; color: white; padding: 0px 5px;
}}
QPushButton#FramelessBtn {{
    background-color: #444; border: none; border-left: 1px solid #555;
    border-top-right-radius: 4px; border-bottom-right-radius: 4px;
    font-weight: bold; color: white;
}}
QPushButton#FramelessBtn:hover {{ background-color: #555; }}

QLineEdit {{ background-color: #3a3a3a; border: 1px solid #555; border-radius: 4px; padding: 0px 5px; color: white; }}
QLineEdit#ResultOutput {{
    background-color: #252525; border: 1px solid #444; color: #aaa; font-weight: bold;
}}

QTextEdit {{ background-color: #1e1e1e; border: 1px solid #3e3e3e; color: #ccc; }}
QSlider::groove:horizontal {{ border: 1px solid #3e3e3e; height: 6px; background: #2a2a2a; margin: 2px 0; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: #1f6aa5; border: 1px solid #1f6aa5; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}

QProgressBar {{
    border: 1px solid #444; border-radius: 4px; text-align: center; color: white; background-color: #2a2a2a;
}}
QProgressBar::chunk {{ background-color: #1f6aa5; width: 10px; }}

QMenu {{ background-color: #3a3a3a; border: 1px solid #555; color: white; }}
QMenu::item {{ padding: 5px 20px; }}
QMenu::item:selected {{ background-color: #1f6aa5; }}
"""

# --- CUSTOM WIDGETS ---

class ModernComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.setView(QListView())
        self.lineEdit().setCursor(Qt.CursorShape.ArrowCursor)
        self.currentTextChanged.connect(self.force_cursor_start)
    def force_cursor_start(self, text): self.lineEdit().setCursorPosition(0)
    def showPopup(self):
        idx = self.findText(self.currentText())
        if idx != -1: self.view().setRowHidden(idx, True)
        super().showPopup()
    def hidePopup(self):
        for i in range(self.count()): self.view().setRowHidden(i, False)
        super().hidePopup()

class MergedPathSelector(QFrame):
    clicked = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MergedPathFrame")
        self.setFixedHeight(34)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.line_edit = QLineEdit()
        self.line_edit.setObjectName("FramelessInput")
        self.line_edit.setPlaceholderText("Pfad zum Bild...")
        self.line_edit.setReadOnly(True)
        self.btn = QPushButton("...")
        self.btn.setObjectName("FramelessBtn")
        self.btn.setFixedSize(40, 34)
        self.btn.clicked.connect(self.clicked.emit)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.btn)
    def setText(self, text):
        self.line_edit.setText(text)
        self.line_edit.setCursorPosition(0)
        self.line_edit.setToolTip(text)
    def text(self): return self.line_edit.text()

class ClickableImageLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
    def mousePressEvent(self, event):
        if self.window(): self.window().setFocus()
        super().mousePressEvent(event)

# --- BACKEND LOGIK HELFER ---
def load_keras_model(model_path):
    try: return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return None

def prepare_image(image_path, target_size):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Bildfehler: {e}")
        return None
    
# Test Keine Rückmeldung
class YoloWorker(QThread):
    finished = pyqtSignal(list, str)  # boxes, error_msg

    def __init__(self, script_path: str, img_path: str, model_path: str):
        super().__init__()
        self.script_path = script_path
        self.img_path = img_path
        self.model_path = model_path

    def run(self):
        try:
            result = subprocess.run(
                [sys.executable, self.script_path, self.img_path, self.model_path],
                capture_output=True,
                text=True
            )
        except Exception as e:
            self.finished.emit([], f"Subprocess Startfehler: {e}")
            return

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if stdout == "":
            msg = f"Kein JSON auf STDOUT.\nReturncode: {result.returncode}\n\nSTDERR:\n{stderr}"
            self.finished.emit([], msg)
            return

        try:
            out = json.loads(stdout)
            boxes = out.get("boxes", [])
            err = out.get("error", "")
            self.finished.emit(boxes, err)
        except Exception as e:
            msg = f"JSON Parse Fehler: {e}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            self.finished.emit([], msg)

# --- HAUPTFENSTER ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QR-Code Detection & Analysis Suite")
        self.resize(1300, 800)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # State
        self.original_pixmap = None
        self.image_list = []      
        self.current_img_idx = 0  
        self.current_model = None
        self.current_model_path = ""
        self.yolo_worker = None
        
        # Cache für Ergebnisse: { 'pfad_zum_bild': score_float }
        self.model_scores = {}

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sidebar
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(460)
        layout.addWidget(self.sidebar)

        sb_layout = QVBoxLayout(self.sidebar)
        sb_layout.setContentsMargins(15, 20, 15, 20)
        sb_layout.setSpacing(15)
        self.create_sidebar(sb_layout)

        # Image Area
        self.image_area = QWidget()
        layout.addWidget(self.image_area)
        
        img_layout = QVBoxLayout(self.image_area)
        self.image_label = ClickableImageLabel("Bitte Bild oder Ordner auswählen")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #444; color: #888; font-size: 20px;")
        img_layout.addWidget(self.image_label)

        self.scan_models()

    def create_sidebar(self, layout):
        layout.addWidget(QLabel("Steuerung", objectName="Header"))

        # 1. MODELL
        layout.addWidget(QLabel("Modell wählen:"))
        self.model_combo = ModernComboBox() 
        layout.addWidget(self.model_combo)

        # 2. BILDQUELLE
        layout.addWidget(QLabel("Bildquelle (Datei oder Ordner):"))
        self.path_selector = MergedPathSelector()
        self.path_selector.clicked.connect(self.show_browse_menu)
        layout.addWidget(self.path_selector)

        # 2b. NAVIGATION
        nav_container = QWidget()
        nav_layout = QHBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_prev = QPushButton("<")
        self.btn_prev.setObjectName("NavBtn")
        self.btn_prev.setFixedSize(40, 30) 
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setEnabled(False)
        self.btn_prev.setFocusPolicy(Qt.FocusPolicy.NoFocus) 
        
        self.lbl_counter = QLabel("0 / 0")
        self.lbl_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_counter.setStyleSheet("color: #888; font-size: 13px;")
        
        self.btn_next = QPushButton(">")
        self.btn_next.setObjectName("NavBtn")
        self.btn_next.setFixedSize(40, 30)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setEnabled(False)
        self.btn_next.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_counter)
        nav_layout.addWidget(self.btn_next)
        layout.addWidget(nav_container)

        # 3. SCHWELLWERT
        layout.addWidget(QLabel("Klassifikations-Schwellwert:"))
        thresh_cont = QWidget()
        t_layout = QHBoxLayout(thresh_cont)
        t_layout.setContentsMargins(0, 0, 0, 0)
        t_layout.setSpacing(10)
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(70)
        self.thresh_slider.valueChanged.connect(self.on_slider_change) 
        self.thresh_input = QLineEdit("0.70")
        self.thresh_input.setFixedWidth(50)
        self.thresh_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val = QDoubleValidator(0.0, 1.0, 2)
        val.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.thresh_input.setValidator(val)
        self.thresh_input.editingFinished.connect(self.sync_slider)
        t_layout.addWidget(self.thresh_slider)
        t_layout.addWidget(self.thresh_input)
        layout.addWidget(thresh_cont)

        # 4. ACTION (KLASSIFIZIERUNG BATCH)
        btn_run = QPushButton("Modell ausführen")
        btn_run.setObjectName("ActionBtn")
        btn_run.setMinimumHeight(40)
        btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_run.clicked.connect(self.run_batch_detection)
        layout.addWidget(btn_run)
        
        # Progress Bar für Batch
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Ergebnis-Feld
        self.class_result = QLineEdit("Noch keine Klassifikation")
        self.class_result.setObjectName("ResultOutput")
        self.class_result.setReadOnly(True)
        self.class_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.class_result)

        self.sep(layout)

        # 5. READER (Platzhalter für YOLO/OpenCV)
        btn_read = QPushButton("QR-Code erkennen und lesen")
        btn_read.setMinimumHeight(30)
        btn_read.clicked.connect(self.run_yolo_detect_only)
        layout.addWidget(btn_read)

        self.qr_output = QTextEdit()
        self.qr_output.setPlaceholderText("Reader Ergebnisse...")
        self.qr_output.setMaximumHeight(80) 
        self.qr_output.setReadOnly(True)
        layout.addWidget(self.qr_output)

        self.sep(layout)

        # 6. WINKEL
        btn_angle = QPushButton("Winkel berechnen")
        btn_angle.setObjectName("SuccessBtn")
        btn_angle.clicked.connect(self.run_plot)
        layout.addWidget(btn_angle)

        self.plot_box = QWidget()
        self.plot_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.plot_box.setStyleSheet("background-color: #000;")
        QVBoxLayout(self.plot_box)
        layout.addWidget(self.plot_box)

    def sep(self, layout):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #3e3e3e;")
        layout.addWidget(line)

    # --- SLIDER LOGIK (Mit Live-Update) ---
    def on_slider_change(self, val):
        self.sync_input(val)
        self.apply_visuals()

    def sync_input(self, val): self.thresh_input.setText(f"{val/100:.2f}")
    def sync_slider(self):
        try:
            v = float(self.thresh_input.text().replace(',', '.'))
            self.thresh_slider.setValue(int(max(0, min(1, v)) * 100))
        except ValueError: pass

    def scan_models(self):
        self.model_combo.clear()
        found = False
        for d in MODEL_DIRS:
            if os.path.exists(d):
                try:
                    for f in os.listdir(d):
                        if f.endswith((".keras", ".h5")):
                            self.model_combo.addItem(os.path.join(d, f))
                            found = True
                except OSError: pass
        if not found: self.model_combo.addItem("Keine Modelle gefunden")
        if found: self.model_combo.lineEdit().setCursorPosition(0)

    # --- BROWSER ---
    def show_browse_menu(self):
        menu = QMenu(self)
        menu.addAction("Einzelne Datei wählen...", self.browse_file)
        menu.addAction("Ganzen Ordner wählen...", self.browse_folder)
        menu.exec(QCursor.pos())

    def browse_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Bild öffnen", "", "Bilder (*.png *.jpg *.jpeg *.bmp)")
        if f:
            self.image_list = [f]
            self.current_img_idx = 0
            self.model_scores = {} # Cache leeren bei neuem Laden
            self.update_image_display()

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Ordner auswählen")
        if folder: self.load_images_from_folder(folder)

    def load_images_from_folder(self, folder_path):
        found = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(VALID_EXTENSIONS): found.append(os.path.join(root, file))
        if found:
            self.image_list = sorted(found)
            self.current_img_idx = 0
            self.model_scores = {} # Cache leeren
            self.update_image_display()
        else:
            self.image_label.setText("Keine Bilder gefunden!")
            self.path_selector.setText(folder_path)

    def update_image_display(self):
        if not self.image_list: return
        path = self.image_list[self.current_img_idx]
        self.path_selector.setText(path)
        
        self.original_pixmap = QPixmap(path)
        
        # Navigation
        total = len(self.image_list)
        self.lbl_counter.setText(f"{self.current_img_idx + 1} / {total}")
        self.btn_prev.setEnabled(total > 1)
        self.btn_next.setEnabled(total > 1)
        
        # Reset Outputs (nur wenn noch kein Ergebnis gecached)
        if path not in self.model_scores:
            self.class_result.setText("Noch keine Klassifikation")
            self.class_result.setStyleSheet("")
            self.display(self.original_pixmap)
        else:
            self.apply_visuals()

        self.qr_output.clear()
        self.setFocus()

    def next_image(self):
        if self.image_list:
            self.current_img_idx = (self.current_img_idx + 1) % len(self.image_list)
            self.update_image_display()

    def prev_image(self):
        if self.image_list:
            self.current_img_idx = (self.current_img_idx - 1) % len(self.image_list)
            self.update_image_display()

    def display(self, pix):
        if not pix or pix.isNull(): return
        w = max(100, self.image_area.width() - 40)
        h = max(100, self.image_area.height() - 40)
        scaled = pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.setStyleSheet("border: none;")

    # --- BATCH DETECTION (Modell auf ALLE Bilder anwenden) ---
    def run_batch_detection(self):
        if not self.image_list: return
        
        # 1. Modell laden
        selected_model = self.model_combo.currentText()
        if "Keine Modelle" in selected_model or not os.path.exists(selected_model):
            self.class_result.setText("Fehler: Modell ungültig")
            return

        # Nur laden wenn nötig
        if self.current_model_path != selected_model:
            self.class_result.setText("Lade Modell...")
            QApplication.processEvents()
            self.current_model = load_keras_model(selected_model)
            self.current_model_path = selected_model
            if not self.current_model: return
        
        # 2. Setup Progress Bar
        total_imgs = len(self.image_list)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total_imgs)
        self.progress_bar.setValue(0)
        
        # 3. Loop über alle Bilder
        for i, img_path in enumerate(self.image_list):
            if i % 5 == 0: QApplication.processEvents()
            
            # Bild verarbeiten
            img_batch = prepare_image(img_path, target_size=IMG_SIZE)
            if img_batch is not None:
                # Predict
                score = self.current_model.predict(img_batch, verbose=0)[0][0]
                self.model_scores[img_path] = score
            
            self.progress_bar.setValue(i + 1)

        # 4. Fertig
        self.progress_bar.setVisible(False)
        self.class_result.setText("Batch-Verarbeitung abgeschlossen.")
        self.apply_visuals()

    def apply_visuals(self):
        """Zeichnet Rahmen basierend auf gecachtem Score und aktuellem Slider."""
        if not self.image_list or not self.original_pixmap: return
        
        current_path = self.image_list[self.current_img_idx]
        
        # Haben wir ein Ergebnis für dieses Bild?
        if current_path not in self.model_scores:
            self.display(self.original_pixmap)
            return
            
        score = self.model_scores[current_path]
        threshold = self.thresh_slider.value() / 100.0
        
        # Zeichnen
        res_pixmap = self.original_pixmap.copy()
        painter = QPainter(res_pixmap)
        
        pen_width = 15
        pen = QPen()
        pen.setWidth(pen_width)

        if score >= threshold:
            pen.setColor(QColor(0, 255, 0)) # Grün
            result_text = f"QR-Code erkannt, Übereinstimmung: {score:.2f}"
            self.class_result.setStyleSheet("QLineEdit#ResultOutput { color: #00ff00; border: 1px solid #00ff00; }")
        else:
            pen.setColor(QColor(255, 0, 0)) # Rot
            result_text = f"Kein QR-Code gefunden, Übereinstimmung: {score:.2f}"
            self.class_result.setStyleSheet("QLineEdit#ResultOutput { color: #ff5555; border: 1px solid #ff5555; }")
            
        painter.setPen(pen)
        painter.drawRect(0, 0, res_pixmap.width(), res_pixmap.height())
        painter.end()
        
        self.display(res_pixmap)
        self.class_result.setText(result_text)

    # --- YOLO MODEL ---
    def run_yolo_detect_only(self):
        if not self.image_list or self.original_pixmap is None:
            return
        
        import os
        project_dir = os.path.dirname(__file__)

        script_path = os.path.join(project_dir, "2-2_yolo_boxes.py")   # <- DEIN Scriptname
        model_path  = os.path.join(project_dir, "models_yolo", "best.pt")

        if not os.path.exists(script_path):
            self.qr_output.setText(f"Script nicht gefunden:\n{script_path}")
            return

        if not os.path.exists(model_path):
            self.qr_output.setText(f"Modell nicht gefunden:\n{model_path}")
            return

        img_path = self.image_list[self.current_img_idx]

        if not os.path.exists(img_path):
            self.qr_output.setText(f"Bild nicht gefunden:\n{img_path}")
            return

        # Script-Pfad absolut
        script_path = os.path.join(os.path.dirname(__file__), "2-2_yolo_boxes.py")  
        if not os.path.exists(script_path):
            #self.qr_output.setText(f"Script: {script_path}\nModel: {model_path}\nImg: {img_path}")
            self.qr_output.setText(f"Script nicht gefunden:\n{script_path}")
            return

        # UI: sofort Feedback geben
        self.qr_output.setText("YOLO läuft... bitte warten.")
        self.setEnabled(False)  # optional: verhindert Mehrfachklicks während Laufzeit

        # Worker starten
        self.yolo_worker = YoloWorker(script_path, img_path, "models_yolo/best.pt")
        self.yolo_worker.finished.connect(self.on_yolo_finished)
        self.yolo_worker.start()

    def on_yolo_finished(self, boxes, err_msg):
        self.qr_output.setText(
            f"YOLO fertig.\nBoxes: {len(boxes)}\n"
            f"Bild: {self.image_list[self.current_img_idx]}"
        )
        self.setEnabled(True)

        if err_msg:
            # err_msg kann auch leer sein bei Erfolg
            self.qr_output.setText(f"YOLO fertig, aber Hinweis:\n{err_msg}")
        else:
            self.qr_output.setText(f"YOLO Detektionen: {len(boxes)}")

        overlay = self.original_pixmap.copy()
        painter = QPainter(overlay)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(6)
        painter.setPen(pen)
        painter.setFont(QFont("Arial", 14))

        for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawText(x1, max(20, y1 - 10), f"det #{i}")

        painter.end()
        self.display(overlay)

        self.yolo_worker = None

    def run_plot(self):
        if self.plot_box.layout().count(): 
            self.plot_box.layout().takeAt(0).widget().deleteLater()
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            ax.plot([1,2,3], [5,2,8], color='#2ea043')
            ax.set_title("Winkel Analyse")
            c = FigureCanvasQTAgg(fig)
            self.plot_box.layout().addWidget(c)
            c.draw()

    def resizeEvent(self, event):
        if self.image_label.pixmap(): pass 
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        if isinstance(self.focusWidget(), QLineEdit):
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key.Key_Left: self.prev_image()
        elif event.key() == Qt.Key.Key_Right: self.next_image()
        else: super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())