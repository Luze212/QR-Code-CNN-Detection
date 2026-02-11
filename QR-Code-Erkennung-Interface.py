import sys
import os
import glob
import numpy as np
import PySide6
import math

# --- PySide6 IMPORTS ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QComboBox, QLineEdit, QFileDialog, 
                             QSlider, QScrollArea, QFrame, QProgressBar, QSizePolicy,
                             QListWidget, QListWidgetItem, QTextEdit)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer, QLocale
from PySide6.QtGui import QPixmap, QImage, QDoubleValidator, QIcon, QKeyEvent, QPainter, QPen, QColor, QFont, QImageReader

# --- MATPLOTLIB ---
import matplotlib
matplotlib.use('QtAgg') 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

import cv2
from ultralytics import YOLO

# --- KONFIGURATION ---
MODEL_DIRS = ["models_tfl", "models_cnn"]
IMG_SIZE = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_QR_MODEL_PATH = os.path.join("models_yolo", "best.pt")

# --- STYLESHEET ---
STYLESHEET = """
QMainWindow { background-color: #2b2b2b; }
QWidget { color: #ffffff; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
QScrollArea { border: none; background-color: #1e1e1e; }
QWidget#SidebarContent { background-color: #1e1e1e; }
QComboBox, QLineEdit, QPushButton { height: 30px; }
QComboBox, QLineEdit, QTextEdit, QListWidget {
    background-color: #333; border: 1px solid #555; border-radius: 4px;
    padding-left: 5px; color: white; selection-background-color: #1f6aa5;
}
QComboBox::drop-down { border: none; background: #444; width: 20px; border-radius: 4px; }
QPushButton { background-color: #3a3a3a; border: 1px solid #555; border-radius: 4px; color: white; font-weight: bold; }
QPushButton:hover { background-color: #4a4a4a; border-color: #666; }
QPushButton:pressed { background-color: #222; }
QPushButton:disabled { background-color: #2a2a2a; color: #555; border-color: #333; }
QPushButton#RunButton { background-color: #1f6aa5; border: none; font-size: 15px; height: 40px; }
QPushButton#RunButton:hover { background-color: #2a7bb6; }
QSlider::groove:horizontal { border: 1px solid #333; height: 6px; background: #222; margin: 2px 0; border-radius: 3px; }
QSlider::handle:horizontal { background: #1f6aa5; border: 1px solid #1f6aa5; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QLabel#ResultLabel { background-color: #222; border: 2px solid #444; border-radius: 6px; padding: 10px; font-weight: bold; font-size: 16px; color: #888; }
QLabel#ResultLabel[status="success"] { border-color: #2ea043; color: #2ea043; background-color: #1a2e1f; }
QLabel#ResultLabel[status="fail"]    { border-color: #da3633; color: #da3633; background-color: #2e1a1a; }
QListWidget::item { padding: 5px; }
QListWidget::item:selected { background-color: #1f6aa5; color: white; }
QLabel#ImageDisplay { background-color: #181818; border: 2px dashed #333; border-radius: 8px; padding: 6px; }
QLabel#ImageDisplay[status="success"] { border: 4px solid #2ea043; background-color: #122215; }
QLabel#ImageDisplay[status="fail"] { border: 4px solid #da3633; background-color: #2e1a1a; }
QLabel#ImageDisplay[status="neutral"] { border: 2px dashed #333; background-color: #181818; }
QWidget#PlotContainer { border: 2px dashed #444; border-radius: 6px; background-color: #1e1e1e; }
QFrame[frameShape="4"] { color: #444; margin-top: 15px; margin-bottom: 15px; }
"""

# =============================================================================
# --- HILFSFUNKTION FÜR KONSISTENTES LADEN ---
# =============================================================================
def load_aligned_image(path):
    """
    Lädt das Bild IMMER so, wie es das GUI anzeigt (EXIF-Rotation beachtet).
    Gibt ein OpenCV-Bild (BGR) zurück.
    """
    if not os.path.exists(path): return None
    reader = QImageReader(path)
    reader.setAutoTransform(True) # Beachtet EXIF
    img = reader.read()
    if img.isNull(): return None
    
    # Konvertierung Qt -> OpenCV
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    arr = np.array(ptr).reshape(h, w, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

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
        print(f"--- Starte Modell aus Pfad: {self.model_path} ---")
        try:
            target_size = (224, 224) 
            if "models_cnn" in self.model_path:
                target_size = (256, 256)
            
            model = tf.keras.models.load_model(self.model_path)
            results = {}
            total = len(self.image_paths)
            
            for i, path in enumerate(self.image_paths):
                try:
                    img = load_img(path, target_size=target_size)
                    x = img_to_array(img)
                    x = x / 255.0
                    x = np.expand_dims(x, axis=0)
                    prediction = model.predict(x, verbose=0)
                    if isinstance(prediction, list): prediction = prediction[0]
                    score = float(prediction[0][0]) if np.ndim(prediction) > 1 else float(prediction[0])
                    results[path] = score
                except Exception as e_img:
                    print(f"Fehler bei Bild {os.path.basename(path)}: {e_img}")
                self.progress.emit(int(((i + 1) / total) * 100))
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

# 2. YOLO WORKER (NUTZT load_aligned_image)
class YoloWorker(QThread):
    progress = Signal(int)
    finished = Signal(list) 
    error = Signal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            abs_model_path = os.path.abspath(YOLO_QR_MODEL_PATH)
            if not os.path.exists(abs_model_path):
                if os.path.exists("best.pt"): abs_model_path = os.path.abspath("best.pt")
            
            if not os.path.exists(abs_model_path):
                self.finished.emit([])
                return

            model = YOLO(abs_model_path)
            
            # KONSISTENTES LADEN (Gedreht wie GUI)
            img = load_aligned_image(self.image_path)
            if img is None: raise ValueError("Bildfehler")

            # --- TILING LOGIK ---
            detections_full = self.scan_full(model, img)
            detections_tiling = self.scan_tiling(model, img)
            all_detections = detections_full + detections_tiling
            final_boxes = self.simple_nms(all_detections, iou_thresh=0.45)

            boxes_out = []
            for i, det in enumerate(final_boxes):
                x1, y1, x2, y2, conf, cls = det
                boxes_out.append({
                    "id": i + 1,
                    "x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1),
                    "confidence": float(conf)
                })

            self.finished.emit(boxes_out)

        except Exception as e:
            print("YOLO ERROR:", e)
            self.finished.emit([])

    def scan_full(self, model, img, imgsz=1280, conf=0.15):
        results = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)
        detections = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf_val = float(box.conf[0].cpu().numpy())
                cls_val = int(box.cls[0].cpu().numpy())
                detections.append([coords[0], coords[1], coords[2], coords[3], conf_val, cls_val])
        return detections

    def scan_tiling(self, model, img, imgsz=1280, conf=0.15, overlap_ratio=0.25):
        h_img, w_img = img.shape[:2]
        tile_w = int(w_img * (0.5 + overlap_ratio/2))
        tile_h = int(h_img * (0.5 + overlap_ratio/2))
        positions = [(0, 0), (w_img - tile_w, 0), (0, h_img - tile_h), (w_img - tile_w, h_img - tile_h)]
        detections = []
        for x_off, y_off in positions:
            crop = img[y_off : y_off+tile_h, x_off : x_off+tile_w]
            results = model.predict(crop, imgsz=imgsz, conf=conf, verbose=False)
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    local = box.xyxy[0].cpu().numpy()
                    gx1, gy1 = local[0] + x_off, local[1] + y_off
                    gx2, gy2 = local[2] + x_off, local[3] + y_off
                    conf_val = float(box.conf[0].cpu().numpy())
                    cls_val = int(box.cls[0].cpu().numpy())
                    detections.append([gx1, gy1, gx2, gy2, conf_val, cls_val])
        return detections

    def simple_nms(self, detections, iou_thresh=0.45):
        if not detections: return []
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [x for x in detections if self.compute_iou(best, x) < iou_thresh]
        return keep

    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        b1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        b2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        return inter / (b1 + b2 - inter + 1e-6)


# 3. READER WORKER (NUTZT load_aligned_image)
class ReaderWorker(QThread):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, image_path, boxes):
        super().__init__()
        self.image_path = image_path
        self.boxes = boxes

    def run(self):
        try:
            results = []
            def _clip_int(v, lo, hi): return max(int(lo), min(int(round(v)), int(hi)))
            def _decode_opencv(img):
                qrd = cv2.QRCodeDetector()
                d, _, _ = qrd.detectAndDecode(img)
                return (True, d.strip()) if d else (False, "")
            
            try: import zxingcpp; _zxing_ok = True
            except: zxingcpp = None; _zxing_ok = False

            def _decode_zxing(img):
                if not _zxing_ok: return False, ""
                try: b = zxingcpp.read_barcodes(img)
                except: b = zxingcpp.read_barcodes(np.ascontiguousarray(img))
                return (True, b[0].text.strip()) if b else (False, "")

            # KONSISTENTES LADEN
            image = load_aligned_image(self.image_path)
            if image is None: raise RuntimeError("Bildfehler")
            H, W = image.shape[:2]

            for box in self.boxes:
                cid = box.get("id", 0)
                x, y = _clip_int(box.get("x",0), 0, W-1), _clip_int(box.get("y",0), 0, H-1)
                w, h = _clip_int(box.get("w",1), 1, W), _clip_int(box.get("h",1), 1, H)
                pad = int(0.08 * max(w, h))
                x0, y0 = max(0, x-pad), max(0, y-pad)
                x1, y1 = min(W, x+w+pad), min(H, y+h+pad)

                if x1<=x0 or y1<=y0: 
                    results.append({"reader":"OpenCV", "code_id":cid, "success":False, "content":""})
                    continue

                roi = image[y0:y1, x0:x1]
                
                # 1. OpenCV
                ok, txt = _decode_opencv(roi)
                results.append({"reader":"OpenCV", "code_id":cid, "success":ok, "content":txt})
                # 2. ZXing
                if _zxing_ok:
                    okz, txtz = _decode_zxing(roi)
                    results.append({"reader":"ZXing", "code_id":cid, "success":okz, "content":txtz})
                else:
                    results.append({"reader":"ZXing", "code_id":cid, "success":False, "content":""})

            self.finished.emit(results)
        except Exception as e: self.error.emit(str(e))


# 4. ANGLE WORKER (NUTZT load_aligned_image)
class AngleWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, image_path, box_id, boxes, fov_x_deg=70.0):
        super().__init__()
        self.image_path = image_path; self.target_id = box_id; self.boxes = boxes; self.fov_x_deg = fov_x_deg

    def run(self):
        try:
            # KONSISTENTES LADEN
            image = load_aligned_image(self.image_path)
            if image is None: raise RuntimeError("Bildfehler")
            H, W = image.shape[:2]
            target = next((b for b in self.boxes if int(b.get("id", -1)) == int(self.target_id)), None)
            if not target: raise RuntimeError("ID nicht gefunden")
            
            x, y, w, h = float(target["x"]), float(target["y"]), float(target["w"]), float(target["h"])
            cx, cy = x + w/2.0, y + h/2.0
            
            fov_x = math.radians(self.fov_x_deg)
            fov_y = 2.0 * math.atan(math.tan(fov_x/2.0)*(H/W))
            yaw = math.radians(math.degrees(((cx - W/2)/W) * fov_x))
            pitch = math.radians(math.degrees(-((cy - H/2)/H) * fov_y))
            rec = np.array([math.sin(yaw)*math.cos(pitch), -math.sin(pitch), math.cos(yaw)*math.cos(pitch)])
            rec = rec / (np.linalg.norm(rec)+1e-9)
            
            self.finished.emit({"yaw_deg":math.degrees(yaw), "pitch_deg":math.degrees(pitch), "view_vec":[0,0,1], "rec_vec":rec.tolist()})
        except Exception as e: self.error.emit(str(e))


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

        # Plot Container erstellen
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

    # --- FILE DIALOGS ---
    def open_file_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Bild wählen", "", "Bilder (*.png *.jpg *.jpeg *.bmp *.webp)")
        if f: 
            self.image_paths = [f]
            self.reset_viewer()
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
                text = f"{os.path.basename(folder)} ({len(self.image_paths)})"
                if len(text) > 30: text = text[:15] + "..." + text[-10:]
                self.input_path.setText(text)
            else: self.lbl_filename.setText("0 Bilder")

    # --- NAVIGATION ---
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
        
        fname = os.path.basename(path)
        if len(fname) > 25: fname = fname[:12] + "..." + fname[-10:]
        self.lbl_filename.setText(f"{self.current_index+1}/{len(self.image_paths)}: {fname}")
        
        self.btn_prev.setEnabled(self.current_index > 0)
        self.btn_next.setEnabled(self.current_index < len(self.image_paths) - 1)

        # WICHTIG: GUI lädt über Qt (EXIF-korrigiert, also hochkant)
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        img = reader.read()
        if img.isNull(): return
        base_pixmap = QPixmap.fromImage(img)
        
        avail_size = self.lbl_image.size() - QSize(12, 12)
        scaled_pixmap = base_pixmap.scaled(avail_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        if path in self.yolo_boxes:
            self.draw_overlays(scaled_pixmap, self.yolo_boxes[path], base_pixmap.size())

        self.lbl_image.setPixmap(scaled_pixmap)
        self.update_cnn_visuals()

    def draw_overlays(self, pixmap, boxes, original_size):
        painter = QPainter(pixmap)
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

                readable = box.get("readable", None)
                if readable is True:
                    pen = QPen(QColor("#00ff00"))  # grün
                elif readable is False:
                    pen = QPen(QColor(255, 40, 40, 140))  # rot
                else:
                    pen = QPen(QColor("#00ff00"))  # default (noch nicht gelesen)

                pen.setWidth(3)
                painter.setPen(pen)

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
        self.yolo_boxes[path] = self._filter_boxes_by_min_conf(boxes)
        self.btn_yolo.setEnabled(True); self.btn_yolo.setText("QR-CODES ORTEN")
        self.prog_yolo.setVisible(False)
        self.update_image_view()
        self.setFocus()
    
    # Liefert den aktuellen Min-Konfidenz-Schwellenwert für YOLO
    def _get_min_conf_value(self) -> float:
        try:
            return float(self.slider_yolo.value()) / 100.0
        except Exception:
            return 0.0

    # Filtert eine Box-Liste so, dass nur Boxen übrig bleiben, die auch bei der aktuellen Slider-Einstellung gezeichnet werden würden.
    def _filter_boxes_by_min_conf(self, boxes): 
        boxes = boxes or []
        thresh = self._get_min_conf_value()
        return [b for b in boxes if float(b.get("confidence", 0.0)) >= thresh]

    # --- READER START ---
    def start_reading(self):
        if not self.image_paths: return
        path = self.image_paths[self.current_index]
        if path not in self.yolo_boxes or not self.yolo_boxes[path]:
            self.txt_content.setText("FEHLER: Bitte erst QR-Codes orten (YOLO)!")
            return
        
        self.btn_read.setEnabled(False); self.btn_read.setText("LESE...")
        self.list_reader.clear()

        boxes_to_read = self._filter_boxes_by_min_conf(self.yolo_boxes.get(path, []))                   
        if not boxes_to_read:                                                                           
            self.txt_content.setText("Keine QR-Boxen über der Min.-Konfidenz. Slider ggf. senken.")     
            self.btn_read.setEnabled(True); self.btn_read.setText("INHALT LESEN")                      
            return                                                                                      

        self.reader_worker = ReaderWorker(path, boxes_to_read)                                         
        self.reader_worker.finished.connect(self.on_read_finished)
        self.reader_worker.start()

    def on_read_finished(self, results):
        self.btn_read.setEnabled(True); self.btn_read.setText("INHALT LESEN")

        # --- pro code_id merken, ob er von irgendeinem Reader lesbar war ---
        readable_by_id = {}
        for res in results:
            cid = int(res.get("code_id", -1))
            ok = bool(res.get("success", False))
            readable_by_id[cid] = readable_by_id.get(cid, False) or ok

        # --- Lesbarkeit in die gespeicherten YOLO-Boxen für das aktuelle Bild schreiben ---
        path = self.image_paths[self.current_index]
        boxes = self.yolo_boxes.get(path, [])
        for b in boxes:
            bid = int(b.get("id", -1))
            if bid in readable_by_id:
                b["readable"] = readable_by_id[bid]
        
        for res in results:
            icon = "✅" if res['success'] else "❌"
            item = QListWidgetItem(f"{icon} {res['reader']} (ID: #{res['code_id']})")
            item.setData(Qt.ItemDataRole.UserRole, res['content'])
            self.list_reader.addItem(item)
        self.update_image_view()  # --- zeichnen, damit Rahmenfarben aktualisiert werden
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
        self.angle_worker = AngleWorker(path, int(target_id_str), self.yolo_boxes.get(path, []), fov_x_deg=70.0)
        self.angle_worker.finished.connect(self.on_angle_finished)
        self.angle_worker.start()

    def on_angle_finished(self, data):  # Winkel-Visualisierung komplette Funktion
         # ---- Setup Figure / Axes ----
        self.plot_figure.clear()
        ax = self.plot_figure.add_subplot(111, projection='3d')
        ax.set_facecolor('#1e1e1e')

        # ---- Farben ----
        col_txt = 'white'
        col_ticks = 'gray'
        col_axes = '#9a9a9a'       # Achsen neutral (kein Default-Blau)

        col_cam = '#0b3d91'        # Kamera: dunkleres Blau
        col_target = '#000000'     # Schwenkrichtung: schwarz

        col_yaw = '#00ff2a'        # sehr sattes Grün
        col_pitch = '#ff0000'      # sattes Rot

        # QR-Ebene sichtbarer
        col_plane_face = (0.15, 0.80, 1.00, 0.28)   # leicht cyan/blau, transparenter Look
        col_plane_edge = (0.15, 0.80, 1.00, 0.75)

        # ---- Achsenbeschriftung (konsistent zum Decken-Plot) ----
        ax.set_xlabel("X: von links nach rechts", color=col_txt)
        ax.set_ylabel("Y: von unten nach oben", color=col_txt)
        ax.set_zlabel("Z: Deckenhöhe", color=col_txt)

        # ---- Achsenlinien ----
        axis_len = 1.0
        axis_w = 1.6
        ax.plot([0, axis_len], [0, 0],       [0, 0],       linewidth=axis_w, color=col_axes)  # +X
        ax.plot([0, 0],       [0, axis_len], [0, 0],       linewidth=axis_w, color=col_axes)  # +Y
        ax.plot([0, 0],       [0, 0],       [0, -axis_len], linewidth=axis_w, color=col_axes) # +Z (down)

        # ---- Hilfsfunktion: Pfeilspitze am Ende einer Kurve ----
        def add_arrowhead(points_xyz, color, size=0.11, width=3):
            t0 = points_xyz[-2]
            t1 = points_xyz[-1]
            d = t1 - t0
            d = d / (np.linalg.norm(d) + 1e-9)

            up = np.array([0.0, 0.0, 1.0])
            s = np.cross(d, up)
            if np.linalg.norm(s) < 1e-6:
                up = np.array([0.0, 1.0, 0.0])
                s = np.cross(d, up)
            s = s / (np.linalg.norm(s) + 1e-9)

            wing1 = t1 - d * size + s * (size * 0.7)
            wing2 = t1 - d * size - s * (size * 0.7)

            ax.plot([t1[0], wing1[0]], [t1[1], wing1[1]], [t1[2], wing1[2]], color=color, linewidth=width)
            ax.plot([t1[0], wing2[0]], [t1[1], wing2[1]], [t1[2], wing2[2]], color=color, linewidth=width)

        # ---- Vektor: Kamera aktuell (aus data["view_vec"]) ----
        v = np.array(data["view_vec"], dtype=float)
        v_plot = np.array([v[0], -v[1], -v[2]], dtype=float)  # Decken-Plot (Y und Z gespiegelt)
        v_plot = v_plot / (np.linalg.norm(v_plot) + 1e-9)

        ax.quiver(
            0, 0, 0, v_plot[0], v_plot[1], v_plot[2],
            color=col_cam, length=1.0, normalize=True,
            linestyle='solid', linewidth=1.4,
            label='Kamera (aktuell)'
        )

        # ---- Schwenkrichtung aus Yaw/Pitch ----
        yaw_deg = float(data["yaw_deg"])
        pitch_deg = float(data["pitch_deg"])
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)

        dir_cam = np.array([
            math.sin(yaw) * math.cos(pitch),
            -math.sin(pitch),
            math.cos(yaw) * math.cos(pitch)
        ], dtype=float)
        dir_cam = dir_cam / (np.linalg.norm(dir_cam) + 1e-9)

        dir_plot = np.array([dir_cam[0], -dir_cam[1], -dir_cam[2]], dtype=float)
        dir_plot = dir_plot / (np.linalg.norm(dir_plot) + 1e-9)

        ax.quiver(
            0, 0, 0, dir_plot[0], dir_plot[1], dir_plot[2],
            color=col_target, length=1.0, normalize=True,
            linewidth=2.2,
            label='Schwenkrichtung (auf QR)'
        )

        # ----------------------------------------------------------------------
        # QR-Ebene als "Zettel/QR-Fläche": an Zielvektor gekoppelt
        # + 0.5 tiefer (also -0.5 zusätzlich)
        # ----------------------------------------------------------------------
        qr_plane_z = -1.2  # vorher -1.0, jetzt 0.5 tiefer

        if abs(dir_plot[2]) > 1e-6:
            t_hit = qr_plane_z / dir_plot[2]
        else:
            t_hit = 1.0

        qr_cx = t_hit * dir_plot[0]
        qr_cy = t_hit * dir_plot[1]
        qr_cz = qr_plane_z

        qr_w, qr_h = 0.9, 0.6

        p1 = (qr_cx - qr_w/2, qr_cy - qr_h/2, qr_cz)
        p2 = (qr_cx + qr_w/2, qr_cy - qr_h/2, qr_cz)
        p3 = (qr_cx + qr_w/2, qr_cy + qr_h/2, qr_cz)
        p4 = (qr_cx - qr_w/2, qr_cy + qr_h/2, qr_cz)

        plane = Poly3DCollection(
            [[p1, p2, p3, p4]],
            facecolor=col_plane_face,
            edgecolor=col_plane_edge,
            linewidths=1.3
        )
        ax.add_collection3d(plane)

        ax.text(
            qr_cx, qr_cy, qr_cz, "QR-Ebene",
            color='white', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=(0, 0, 0, 0.65), edgecolor='none')
        )

        # ----------------------------------------------------------------------
        # Referenz-Geraden (beschriftet, sehr gut lesbar)
        # ----------------------------------------------------------------------
        ax.plot([0.0, -1.5], [0.0, 0.0], [0.0, 0.0], color=col_pitch, linewidth=3)
        ax.text(
            -1.5, 0.0, 0.0, "Pitch",
            color=col_txt, fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.28', facecolor=col_pitch, edgecolor='none', alpha=0.92)
        )

        ax.plot([0.0, 0.0], [0.0, 1.5], [0.0, 0.0], color=col_yaw, linewidth=3)
        ax.text(
            0.0, 1.5, 0.0, "Yaw",
            color=col_txt, fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.28', facecolor=col_yaw, edgecolor='none', alpha=0.92)
        )

        # ---- Referenz-Rotationsbögen (160°) ----
        arc_r = 0.40
        yaw_off = np.array([0.0, 1.0, 0.0])
        pitch_off = np.array([-1.0, 0.0, 0.0])

        ref_deg = 160.0
        half = math.radians(ref_deg / 2.0)
        theta_ref = np.linspace(-half, half, 140)

        x = arc_r * np.sin(theta_ref)
        y = np.zeros_like(theta_ref)
        z = -arc_r * np.cos(theta_ref)
        x += yaw_off[0]; y += yaw_off[1]; z += yaw_off[2]
        ax.plot(x, y, z, color=col_yaw, linewidth=3.6)
        add_arrowhead(np.vstack([x, y, z]).T, color=col_yaw, width=5)

        x = np.zeros_like(theta_ref)
        y = -arc_r * np.sin(theta_ref)
        z = -arc_r * np.cos(theta_ref)
        x += pitch_off[0]; y += pitch_off[1]; z += pitch_off[2]
        ax.plot(x, y, z, color=col_pitch, linewidth=3.6)
        add_arrowhead(np.vstack([x, y, z]).T, color=col_pitch, width=5)

        # ---- Titel / Ticks / Limits ----
        ax.set_title(f"Yaw: {yaw_deg:.1f}° | Pitch: {pitch_deg:.1f}°", color=col_txt)
        ax.tick_params(colors=col_ticks)

        ax.set_xlim([-1.7, 1.3])
        ax.set_ylim([-1.3, 1.7])
        ax.set_zlim([-1.6, 0.0])

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