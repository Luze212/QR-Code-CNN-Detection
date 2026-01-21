import customtkinter as ctk 
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- KONFIGURATION ---
ctk.set_appearance_mode("Dark")  # System, Dark, Light
ctk.set_default_color_theme("dark-blue")  # Themes: blue, dark-blue, green

class QRCodeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Fenster Konfiguration
        self.title("QR-Code Detection & Analysis Suite")
        self.geometry("1200x800")
        
        # Grid Layout (1 Zeile, 2 Spalten)
        # Spalte 0 (Links): Steuerung (Gewichtung 1)
        # Spalte 1 (Rechts): Bildanzeige (Gewichtung 3 -> ca. 3/4)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # --- LINKE SEITE (STEUERUNG) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1) # Leerraumfüller unten

        self.create_sidebar_content()

        # --- RECHTE SEITE (BILDANZEIGE) ---
        self.image_area_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.image_area_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        self.create_image_area()
        
        # Initialisierung
        self.loaded_image_path = None
        self.scan_models()

    def create_sidebar_content(self):
        """Erstellt alle Elemente der linken Steuerungsleiste"""
        
        # 1. Überschrift
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Steuerung", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # 2. Modell Auswahl (Dropdown)
        self.model_label = ctk.CTkLabel(self.sidebar_frame, text="Modell wählen:", anchor="w")
        self.model_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.model_option_menu = ctk.CTkOptionMenu(self.sidebar_frame, dynamic_resizing=False,
                                                   values=["Keine Modelle gefunden"])
        self.model_option_menu.grid(row=2, column=0, padx=20, pady=(5, 10), sticky="ew")

        # 3. Bildauswahl (Pfad + Button)
        self.path_label = ctk.CTkLabel(self.sidebar_frame, text="Bildquelle:", anchor="w")
        self.path_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        
        path_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        path_frame.grid(row=4, column=0, padx=20, pady=(5, 10), sticky="ew")
        
        self.path_entry = ctk.CTkEntry(path_frame, placeholder_text="Pfad zum Bild...")
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.browse_button = ctk.CTkButton(path_frame, text="...", width=40, command=self.browse_image)
        self.browse_button.pack(side="right")

        # --- SEPARATOR ---
        ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="gray30").grid(row=5, column=0, sticky="ew", padx=20, pady=10)

        # 4. QR-Code Reader Bereich
        self.reader_btn = ctk.CTkButton(self.sidebar_frame, text="QR-Reader ausführen", command=self.run_qr_backend)
        self.reader_btn.grid(row=6, column=0, padx=20, pady=10, sticky="ew")
        
        # Tabelle für Reader Ergebnisse
        self.reader_frame = ctk.CTkScrollableFrame(self.sidebar_frame, height=100, label_text="Reader Status")
        self.reader_frame.grid(row=7, column=0, padx=20, pady=5, sticky="ew")
        
        # Beispiel Einträge (Platzhalter für später)
        self.reader_rows = []
        for reader_name in ["OpenCV", "PyZbar", "ZXing"]:
            row_frame = ctk.CTkFrame(self.reader_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)
            lbl = ctk.CTkLabel(row_frame, text=reader_name, anchor="w")
            lbl.pack(side="left")
            status = ctk.CTkLabel(row_frame, text="---", text_color="gray")
            status.pack(side="right")
            self.reader_rows.append((lbl, status))

        # Ergebnisfeld (Inhalt)
        self.qr_content = ctk.CTkTextbox(self.sidebar_frame, height=60)
        self.qr_content.grid(row=8, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.qr_content.insert("0.0", "QR-Inhalt erscheint hier...")
        self.qr_content.configure(state="disabled")

        # --- SEPARATOR ---
        ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="gray30").grid(row=9, column=0, sticky="ew", padx=20, pady=10)

        # 5. Kamerawinkel & Plot
        self.angle_btn = ctk.CTkButton(self.sidebar_frame, text="Winkel berechnen", 
                                       fg_color="green", hover_color="darkgreen",
                                       command=self.run_angle_backend)
        self.angle_btn.grid(row=10, column=0, padx=20, pady=10, sticky="ew")
        
        # Frame für Matplotlib
        self.plot_container = ctk.CTkFrame(self.sidebar_frame, fg_color="black")
        self.plot_container.grid(row=11, column=0, padx=20, pady=(5, 20), sticky="nsew")
        # Wir geben diesem Bereich Gewicht, damit er den Rest füllt
        self.sidebar_frame.grid_rowconfigure(11, weight=1)

    def create_image_area(self):
        """Erstellt den rechten Bereich zur Bildanzeige"""
        self.image_label = ctk.CTkLabel(self.image_area_frame, text="Bitte Bild auswählen\nund Modell starten",
                                        font=ctk.CTkFont(size=24))
        self.image_label.pack(expand=True, fill="both")

    # --- HILFSFUNKTIONEN GUI ---

    def scan_models(self):
        """Durchsucht den 'models' Ordner und füllt das Dropdown"""
        models_dir = "models"
        model_list = []
        
        if os.path.exists(models_dir):
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(".keras") or file.endswith(".h5"):
                        # Relativen Pfad als Kategorie nutzen (z.B. "tuned/model.keras")
                        rel_path = os.path.relpath(os.path.join(root, file), models_dir)
                        model_list.append(rel_path)
        
        if model_list:
            self.model_option_menu.configure(values=model_list)
            self.model_option_menu.set(model_list[0])
        else:
            self.model_option_menu.configure(values=["Keine Modelle gefunden"])

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Bilder", "*.jpg *.png *.jpeg")])
        if file_path:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, file_path)
            self.loaded_image_path = file_path
            self.display_image(file_path)

    def display_image(self, path):
        """Zeigt das Bild rechts an (Skaliert)"""
        try:
            pil_image = Image.open(path)
            # Skalieren für die Anzeige (Max 800x800 z.B.)
            w, h = pil_image.size
            aspect = w / h
            target_h = 600
            target_w = int(target_h * aspect)
            
            ctk_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(target_w, target_h))
            self.image_label.configure(image=ctk_img, text="")
        except Exception as e:
            print(f"Fehler beim Laden des Bildes: {e}")

    # --- BACKEND PLATZHALTER (Deine Logik hier später einfügen) ---

    def _backend_crop_and_zoom(self, bounding_box):
        """
        PLATZHALTER: Schneidet den Bereich aus dem Bild aus und vergrößert ihn.
        bounding_box: Koordinaten vom Modell (x, y, w, h)
        return: Das zugeschnittene Bild
        """
        # TODO: Implementiere hier das Zuschneiden basierend auf Model-Prediction
        print("Backend: Bild wird zugeschnitten und optimiert...")
        pass

    def run_qr_backend(self):
        """
        Wird ausgeführt, wenn 'QR-Reader ausführen' geklickt wird.
        Hier kommt die Logik rein, die:
        1. Das Modell auf das Bild anwendet (Bounding Box finden)
        2. _backend_crop_and_zoom aufruft
        3. Die Reader (OpenCV, etc.) auf das Crop anwendet
        """
        print("Backend: Starte QR-Code Analyse...")
        
        # Simulation eines Ergebnisses
        found_something = True 
        
        if found_something:
            # Beispielhaftes Update der GUI Tabelle
            self.reader_rows[0][1].configure(text="TREFFER", text_color="green") # OpenCV
            self.reader_rows[1][1].configure(text="FEHLER", text_color="red")    # PyZbar
            
            # Ergebnis Text setzen
            self.qr_content.configure(state="normal")
            self.qr_content.delete("0.0", "end")
            self.qr_content.insert("0.0", "https://www.beispiel-qr.de/waschanlage/id=123")
            self.qr_content.configure(state="disabled")
            
            # TODO: Hier Zeichnen wir später den Rahmen ins Bild rechts
            # self.draw_bounding_box_on_display(...)

    def run_angle_backend(self):
        """
        Wird ausgeführt, wenn 'Winkel berechnen' geklickt wird.
        Hier kommt die Logik für die Verzerrungsberechnung rein.
        """
        print("Backend: Berechne optimalen Winkel...")
        
        # --- PLOT PLATZHALTER ---
        # Wir erstellen einen Dummy-Plot für die GUI
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        
        # Darkmode Style für Plot
        fig.patch.set_facecolor('#2b2b2b') # Hintergrund Farbe dunkelgrau
        ax.set_facecolor('#2b2b2b')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white') 
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Dummy Daten
        x = [10, 20, 30, 40] # Winkel
        y = [0.2, 0.4, 0.8, 0.6] # Lesbarkeit
        ax.plot(x, y, color='cyan', marker='o')
        ax.set_title("Lesbarkeit vs. Winkel", color='white', fontsize=8)

        # Alten Plot entfernen falls vorhanden
        for widget in self.plot_container.winfo_children():
            widget.destroy()

        # Canvas in Tkinter einbetten
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    app = QRCodeApp()
    app.mainloop()