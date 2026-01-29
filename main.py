import sys
import os
import json
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QFileDialog, QLabel, QSlider,
                             QDialog, QFormLayout, QDoubleSpinBox, QSpinBox, 
                             QComboBox, QDialogButtonBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMessageBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QColor
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# IMPORTACIONES DE NUESTROS MODULOS
from gui_components import DetectionBar
from detector import AnalysisThread

# --- 1. MODAL DE CONFIGURACIÓN ---
class SettingsModal(QDialog):
    def __init__(self, current_config, model_classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración Avanzada por Clase")
        self.setMinimumWidth(600)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 20)
        self.stride_spin.setValue(current_config.get('stride', 1))
        form_layout.addRow("Saltar Frames (Stride):", self.stride_spin)
        
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(current_config.get('iou', 0.45))
        form_layout.addRow("IoU (Superposición):", self.iou_spin)
        layout.addLayout(form_layout)

        layout.addWidget(QLabel("\nParámetros individuales por clase:"))
        
        self.table = QTableWidget(len(model_classes), 3) 
        self.table.setHorizontalHeaderLabels(["Clase Original", "Etiqueta (Alias)", "Confianza Mín."])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 200)
        self.table.setColumnWidth(1, 150)
        
        self.class_map = current_config.get('class_map', {})

        for i, name in enumerate(model_classes):
            item_name = QTableWidgetItem(name)
            item_name.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, 0, item_name)

            current_alias = self.class_map.get(name, {}).get('alias', name[:2].upper())
            self.table.setItem(i, 1, QTableWidgetItem(current_alias))

            spin_conf = QDoubleSpinBox()
            spin_conf.setRange(0.01, 1.0)
            spin_conf.setSingleStep(0.05)
            current_conf = self.class_map.get(name, {}).get('conf', 0.40)
            spin_conf.setValue(current_conf)
            
            widget = QWidget()
            layout_spin = QHBoxLayout(widget)
            layout_spin.addWidget(spin_conf)
            layout_spin.setAlignment(Qt.AlignCenter)
            layout_spin.setContentsMargins(0,0,0,0)
            self.table.setCellWidget(i, 2, widget)

        layout.addWidget(self.table)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self):
        new_map = {}
        for i in range(self.table.rowCount()):
            original_name = self.table.item(i, 0).text()
            alias = self.table.item(i, 1).text()
            widget = self.table.cellWidget(i, 2)
            spinbox = widget.findChild(QDoubleSpinBox)
            conf_val = spinbox.value()
            new_map[original_name] = {'alias': alias, 'conf': conf_val, 'active': True}
        return {'stride': self.stride_spin.value(), 'iou': self.iou_spin.value(), 'class_map': new_map}

# --- 2. MODAL DE RESULTADOS (% ACIERTO) ---
class ResultsModal(QDialog):
    def __init__(self, truth_bar, detection_bars, total_frames, current_stride, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Análisis de Cobertura (Ajustado a Stride: {current_stride})")
        self.setMinimumSize(700, 350)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Clase", "Frames Reales (Muestreados)", "Aciertos (TP)", "Cobertura Real"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        truth_buffer = truth_bar.buffer
        truth_sampled_indices = [i for i in range(0, total_frames, current_stride) 
                                 if i < len(truth_buffer) and truth_buffer[i] > 0]
        total_truth_sampled = len(truth_sampled_indices)

        row = 0
        for name, bar in detection_bars.items():
            ai_buffer = bar.buffer
            limit = min(len(truth_buffer), len(ai_buffer), total_frames)
            valid_intersection = 0 

            for i in range(0, limit, current_stride):
                if truth_buffer[i] > 0 and ai_buffer[i] > 0:
                    valid_intersection += 1

            if total_truth_sampled > 0:
                pct = (valid_intersection / total_truth_sampled * 100)
            else:
                pct = 0.0

            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(str(total_truth_sampled)))
            self.table.setItem(row, 2, QTableWidgetItem(str(valid_intersection)))
            
            item_pct = QTableWidgetItem(f"{pct:.2f}%")
            if pct >= 90: item_pct.setForeground(Qt.green)
            elif pct < 50: item_pct.setForeground(Qt.red)
            else: item_pct.setForeground(QColor("orange"))
            self.table.setItem(row, 3, item_pct)
            row += 1

        layout.addWidget(QLabel(f"Nota: Cálculo muestreando 1 de cada {current_stride} frames."))
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

# --- 3. MODAL DE MÉTRICAS (KAPPA) ---
class MetricsModal(QDialog):
    def __init__(self, truth_bar, detection_bars, total_frames, stride, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Informe Técnico (Stride: {stride})")
        self.setMinimumSize(800, 400)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["Clase", "Kappa Cohen", "Recall", "Precision", "TP", "FP", "FN"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        y_true_raw = truth_bar.buffer[:total_frames]
        y_true = y_true_raw[::stride] 

        row = 0
        for name, bar in detection_bars.items():
            if len(bar.buffer) < total_frames: continue
            y_pred = bar.buffer[:total_frames][::stride]

            try:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
            except:
                tn, fp, fn, tp = 0, 0, 0, 0

            kappa = cohen_kappa_score(y_true, y_pred)
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            
            item_kappa = QTableWidgetItem(f"{kappa:.3f}")
            if kappa > 0.8: item_kappa.setForeground(Qt.green)
            elif kappa < 0.4: item_kappa.setForeground(Qt.red)
            self.table.setItem(row, 1, item_kappa)

            self.table.setItem(row, 2, QTableWidgetItem(f"{recall:.1%}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{precision:.1%}"))
            self.table.setItem(row, 4, QTableWidgetItem(str(tp)))
            self.table.setItem(row, 5, QTableWidgetItem(str(fp)))
            self.table.setItem(row, 6, QTableWidgetItem(str(fn)))
            row += 1

        layout.addWidget(QLabel(f"Nota: Métricas calculadas con Stride {stride}."))
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

# --- 4. MODAL DE LOG DE EVENTOS (NUEVA LÓGICA SOLICITADA) ---
class EventsLogModal(QDialog):
    def __init__(self, video_filename, json_data_full, detection_bars, fps, total_frames, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Desglose de Frames por Evento: {video_filename}")
        self.setMinimumSize(1100, 600)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "ID Evento", "Clase JSON", "Inicio", "Duración (Frames)", 
            "DETECCIONES POR CLASE (Frames Captados)", "Estado"
        ])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch) # Columna de detecciones ancha
        layout.addWidget(self.table)

        video_data = json_data_full.get(video_filename)
        if not video_data:
            layout.addWidget(QLabel(f"❌ Error: No hay datos en JSON para: {video_filename}"))
            return

        row = 0
        v_fps = video_data.get("fps", fps)
        
        # Iterar sobre los eventos del JSON (Ground Truth)
        for tap in video_data.get("taps", []):
            json_class_name = str(tap.get("tap_id", "Desconocido"))
            
            for i, cycle in enumerate(tap.get("cycles", [])):
                self.table.insertRow(row)
                
                # --- CALCULAR INTERVALO DE TIEMPO (FRAMES) ---
                start_sec = cycle.get("start_time_sec", 0)
                end_sec = cycle.get("end_time_sec", 0)
                
                f_start = int(start_sec * v_fps)
                f_end = int(end_sec * v_fps)
                f_start = max(0, f_start)
                f_end = min(total_frames - 1, f_end)
                
                duration_frames = f_end - f_start
                if duration_frames <= 0: continue

                # --- LÓGICA DE RECUENTO: MIRAR TODAS LAS CLASES ---
                # "veamos el n de frames captados por la ia... en todas las clases"
                detections_summary = []
                total_detected_frames_any_class = 0

                # Recorremos todas las barras de la IA disponibles
                for ai_name, ai_bar in detection_bars.items():
                    # Extraemos el fragmento de memoria correspondiente a este evento
                    fragment = ai_bar.buffer[f_start:f_end+1]
                    
                    # Contamos cuántos frames (1s) hay en este fragmento
                    frames_count = sum(1 for x in fragment if x > 0)
                    
                    if frames_count > 0:
                        # Formato: "Clase: N frames"
                        detections_summary.append(f"{ai_name}: {frames_count}")
                        total_detected_frames_any_class += frames_count

                # Construimos el string para la celda
                if detections_summary:
                    summary_str = " | ".join(detections_summary)
                    status = "✅ CON DATOS"
                    color = Qt.green
                else:
                    summary_str = "--- (Silencio Total) ---"
                    status = "❌ VACÍO"
                    color = Qt.red

                # --- PINTAR LA FILA ---
                self.table.setItem(row, 0, QTableWidgetItem(f"#{i+1}"))
                self.table.setItem(row, 1, QTableWidgetItem(json_class_name))
                self.table.setItem(row, 2, QTableWidgetItem(self.format_seconds(start_sec)))
                
                # Duración en frames (para tener contexto del número de detecciones)
                self.table.setItem(row, 3, QTableWidgetItem(f"{duration_frames} f"))
                
                # LA COLUMNA IMPORTANTE: Desglose
                item_summary = QTableWidgetItem(summary_str)
                item_summary.setToolTip(summary_str)
                self.table.setItem(row, 4, item_summary)
                
                item_status = QTableWidgetItem(status)
                item_status.setForeground(color)
                item_status.setFont(self.get_bold_font())
                self.table.setItem(row, 5, item_status)

                row += 1

        layout.addWidget(QLabel(f"Total Eventos JSON: {row}. La columna central muestra cuántos frames detectó cada clase de la IA en ese periodo."))
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def format_seconds(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def get_bold_font(self):
        f = self.font()
        f.setBold(True)
        return f

# --- APLICACIÓN PRINCIPAL ---
class BeerAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Validador de Bebidas IA - Gamb00za 2026")
        self.resize(1200, 900)
        self.setStyleSheet("QMainWindow { background-color: #121212; } QPushButton { padding: 8px; color: white; background-color: #333; }")
        
        self.video_path = None
        self.model_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 15.0
        self.model_classes = []
        self.detection_bars = {} 
        self.ai_config = {'conf': 0.40, 'stride': 1, 'class_map': {}}
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.analysis_thread = None
        self.video_queue = []
        self.video_root_folder = ""
        self.is_batch_running = False
        self.json_data_per_video = {}

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Fila Archivos
        files_layout = QHBoxLayout()
        self.btn_folder = QPushButton("1. 📁 Carpeta Videos")
        self.btn_folder.clicked.connect(self.select_folder)
        self.btn_vid = QPushButton("2. 📂 Cargar 1 Video")
        self.btn_vid.clicked.connect(self.load_video)
        self.btn_mod = QPushButton("3. 🧠 Cargar Modelo")
        self.btn_mod.clicked.connect(self.load_model)
        self.btn_json = QPushButton("4. 📄 Cargar JSON")
        self.btn_json.clicked.connect(self.load_json)
        for b in [self.btn_folder, self.btn_vid, self.btn_mod, self.btn_json]: files_layout.addWidget(b)
        layout.addLayout(files_layout)

        # Fila Ejecución
        action_layout = QHBoxLayout()
        self.btn_settings = QPushButton("⚙ Ajustar IA")
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_run = QPushButton("▶ EJECUTAR VIDEO ÚNICO")
        self.btn_run.setStyleSheet("background-color: #2e7d32; font-weight: bold;")
        self.btn_run.clicked.connect(self.start_analysis)
        self.btn_run_batch = QPushButton("⏩ EJECUTAR LOTE")
        self.btn_run_batch.setStyleSheet("background-color: #d35400; font-weight: bold;")
        self.btn_run_batch.setEnabled(False) 
        self.btn_run_batch.clicked.connect(self.start_batch_processing)
        action_layout.addWidget(self.btn_settings)
        action_layout.addWidget(self.btn_run)
        action_layout.addWidget(self.btn_run_batch)
        layout.addLayout(action_layout)
        
        # Fila Resultados
        results_layout = QHBoxLayout()
        self.btn_results = QPushButton("📊 Ver % Aciertos")
        self.btn_results.setStyleSheet("background-color: #007acc; font-weight: bold;")
        self.btn_results.clicked.connect(self.show_results)

        self.btn_metrics = QPushButton("📈 Ver Métricas (Kappa)")
        self.btn_metrics.setStyleSheet("background-color: #8e44ad; font-weight: bold;")
        self.btn_metrics.clicked.connect(self.show_metrics)
        
        # --- AQUÍ ESTABA EL ERROR: AHORA DEFINIMOS EL BOTÓN ANTES DE AÑADIRLO ---
        self.btn_log = QPushButton("📋 Log Eventos")
        self.btn_log.setStyleSheet("background-color: #7f8c8d; font-weight: bold;")
        self.btn_log.clicked.connect(self.show_event_log)
        
        results_layout.addWidget(self.btn_results)
        results_layout.addWidget(self.btn_metrics)
        results_layout.addWidget(self.btn_log) # Ahora ya existe
        layout.addLayout(results_layout)

        # Monitor
        self.video_display = QLabel("Seleccione Carpeta, Modelo y JSON")
        self.video_display.setStyleSheet("background: black; border: 1px solid #444;")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumHeight(500)
        layout.addWidget(self.video_display)

        # Controles
        play_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_video)
        self.timeline_bar = QSlider(Qt.Horizontal)
        self.timeline_bar.sliderMoved.connect(self.set_video_position)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("color: white; font-family: monospace;")
        play_layout.addWidget(self.btn_play)
        play_layout.addWidget(self.timeline_bar)
        play_layout.addWidget(self.lbl_time)
        layout.addLayout(play_layout)

        self.bars_container = QVBoxLayout() 
        layout.addLayout(self.bars_container)
        self.bar_truth = DetectionBar("#f1c40f", "TRUTH")
        layout.addWidget(self.bar_truth)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta con Videos")
        if folder:
            self.video_root_folder = folder
            self.btn_folder.setText(f"📁 .../{os.path.basename(folder)}")
            self.check_batch_ready()

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Video", "", "Videos (*.mp4 *.avi)")
        if path: self.load_video_from_path(path)

    def load_video_from_path(self, path):
        if self.cap: self.cap.release()
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.timeline_bar.setRange(0, self.total_frames - 1)
        for bar in self.detection_bars.values(): bar.init_buffer(self.total_frames)
        self.bar_truth.init_buffer(self.total_frames)
        self.set_video_position(0)
        self.setWindowTitle(f"Validador IA - {os.path.basename(path)}")
        
        # Auto-cargar verdad si hay JSON
        if self.json_data_per_video:
            fname = os.path.basename(path)
            # Lógica simple de búsqueda
            if fname not in self.json_data_per_video:
                name_no_ext = os.path.splitext(fname)[0]
                if name_no_ext in self.json_data_per_video: fname = name_no_ext
            self.load_truth_for_current_video(fname)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar .pt", "", "YOLO Model (*.pt)")
        if path:
            self.model_path = path
            try:
                temp_model = YOLO(path)
                self.model_classes = list(temp_model.names.values())
                class_map = {}
                for name in self.model_classes:
                    class_map[name] = {'alias': name[:2].upper(), 'conf': 0.40, 'active': True}
                self.ai_config['class_map'] = class_map
                self.refresh_bars()
                self.btn_mod.setText(f"🧠 {os.path.basename(path)}")
            except Exception as e: print(f"Error modelo: {e}")

    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "JSON Truth", "", "JSON (*.json)")
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.json_data_per_video = {}
                    videos_list = data.get("videos", [])
                    for v in videos_list:
                        fname = ""
                        raw_path = v.get("video_path", "")
                        if raw_path:
                            fname = os.path.basename(raw_path)
                            if "\\" in fname: fname = fname.split("\\")[-1]
                        if not fname: fname = v.get("filename") or v.get("name")
                        
                        if fname:
                            self.json_data_per_video[fname] = v
                            name_no_ext = os.path.splitext(fname)[0]
                            if name_no_ext != fname: self.json_data_per_video[name_no_ext] = v
                    
                    self.btn_json.setText(f"📄 JSON ({len(videos_list)})")
                    if self.video_path:
                        self.load_video_from_path(self.video_path) # Recargar para pintar verdad
                    self.check_batch_ready()
            except Exception as e: print(f"Error JSON: {e}")

    def check_batch_ready(self):
        if self.video_root_folder and self.json_data_per_video:
            self.btn_run_batch.setEnabled(True)

    def refresh_bars(self):
        for i in reversed(range(self.bars_container.count())): 
            item = self.bars_container.itemAt(i)
            if item.widget(): item.widget().setParent(None)
        self.detection_bars.clear()
        colors = ["#4a90e2", "#50e3c2", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]
        class_map = self.ai_config.get('class_map', {})
        for i, (original_name, settings) in enumerate(class_map.items()):
            if settings.get('active', True):
                color = colors[i % len(colors)]
                alias = settings.get('alias', original_name[:2].upper())
                new_bar = DetectionBar(color, alias)
                self.bars_container.addWidget(new_bar)
                self.detection_bars[original_name] = new_bar
                if self.total_frames > 0: new_bar.init_buffer(self.total_frames)

    def open_settings(self):
        if not self.model_classes: return
        modal = SettingsModal(self.ai_config, self.model_classes, self)
        if modal.exec():
            self.ai_config = modal.get_config()
            self.refresh_bars()

    def start_batch_processing(self):
        if not self.model_path: return
        self.video_queue = list(self.json_data_per_video.keys())
        # Filtrar duplicados (por tener nombre con y sin extension)
        self.video_queue = list(set([v.get("filename", k) for k,v in self.json_data_per_video.items() if "filename" in v or "video_path" in v]))
        # Simplificación: Usar las claves directas del dict, filtrando las redundantes si es necesario
        # Para simplificar, usamos todas las claves que parezcan archivos
        self.video_queue = [k for k in self.json_data_per_video.keys() if "." in k] # Truco simple
        self.is_batch_running = True
        self.process_next_in_queue()

    def process_next_in_queue(self):
        if not self.video_queue:
            self.is_batch_running = False
            QMessageBox.information(self, "Fin", "Lote procesado.")
            return
        video_filename = self.video_queue.pop(0)
        full_path = os.path.join(self.video_root_folder, video_filename)
        if not os.path.exists(full_path):
            print(f"Skip {video_filename}")
            self.process_next_in_queue()
            return
        self.load_video_from_path(full_path)
        self.start_analysis(auto_batch=True)

    def load_truth_for_current_video(self, filename):
        video_data = self.json_data_per_video.get(filename)
        if not video_data: return
        self.bar_truth.init_buffer(self.total_frames)
        v_fps = video_data.get("fps", self.fps)
        for tap in video_data.get("taps", []):
            for cycle in tap.get("cycles", []):
                f_start = int(cycle["start_time_sec"] * v_fps)
                f_end = int(cycle["end_time_sec"] * v_fps)
                f_start = max(0, f_start)
                f_end = min(self.total_frames - 1, f_end)
                for f in range(f_start, f_end + 1): self.bar_truth.mark_detection(f, True)
        self.bar_truth.update()

    def start_analysis(self, auto_batch=False):
        if self.video_path and self.model_path:
            current_f = 0 if auto_batch else self.timeline_bar.value()
            if self.analysis_thread and self.analysis_thread.isRunning():
                self.analysis_thread.stop()
                self.analysis_thread.wait()
            if self.timer.isActive(): self.toggle_video()
            self.analysis_thread = AnalysisThread(self.video_path, self.model_path, self.ai_config, current_f)
            self.analysis_thread.frame_ready.connect(self.display_analysis_frame)
            self.analysis_thread.detection_event.connect(self.update_live_bars)
            if auto_batch: self.analysis_thread.finished.connect(self.process_next_in_queue)
            self.analysis_thread.start()

    def update_live_bars(self, idx, detections_dict):
        for class_name, found in detections_dict.items():
            if class_name in self.detection_bars:
                bar = self.detection_bars[class_name]
                bar.mark_detection(idx, found)
                bar.set_current_frame(idx)
        self.bar_truth.set_current_frame(idx)
        self.timeline_bar.setValue(idx)
        self.update_time_label(idx)

    def display_analysis_frame(self, img):
        pixmap = QPixmap.fromImage(img)
        if not pixmap.isNull():
            self.video_display.setPixmap(pixmap.scaled(
                self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_video_position(self, pos):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = self.cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
                self.display_analysis_frame(img)
                for bar in self.detection_bars.values(): bar.set_current_frame(pos)
                self.bar_truth.set_current_frame(pos)
                self.update_time_label(pos)
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.seek(pos)

    def toggle_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶ Play")
            if self.analysis_thread: self.analysis_thread.set_paused(True)
        else:
            self.timer.start(int(1000/self.fps))
            self.btn_play.setText("⏸ Pausa")
            if self.analysis_thread:
                self.analysis_thread.seek(self.timeline_bar.value())
                self.analysis_thread.set_paused(False)

    def next_frame(self):
        curr = self.timeline_bar.value()
        if curr < self.total_frames - 1:
            self.timeline_bar.setValue(curr + 1)
            self.set_video_position(curr + 1)
        else: self.timer.stop()

    def show_results(self):
        if self.total_frames > 0:
            stride = self.ai_config.get('stride', 1)
            modal = ResultsModal(self.bar_truth, self.detection_bars, self.total_frames, stride, self)
            modal.exec()

    def show_metrics(self):
        if self.total_frames > 0:
            stride = self.ai_config.get('stride', 1)
            modal = MetricsModal(self.bar_truth, self.detection_bars, self.total_frames, stride, self)
            modal.exec()

    def show_event_log(self):
        if not self.video_path or not self.json_data_per_video:
            QMessageBox.warning(self, "Faltan datos", "Carga un video y el JSON primero.")
            return
        current_filename = os.path.basename(self.video_path)
        if current_filename not in self.json_data_per_video:
             name_no_ext = os.path.splitext(current_filename)[0]
             if name_no_ext in self.json_data_per_video: current_filename = name_no_ext
        
        modal = EventsLogModal(current_filename, self.json_data_per_video, self.detection_bars, self.fps, self.total_frames, self)
        modal.exec()

    def update_time_label(self, pos):
        cur_s = int(pos / self.fps) if self.fps > 0 else 0
        tot_s = int(self.total_frames / self.fps) if self.fps > 0 else 0
        self.lbl_time.setText(f"{self.format_time(cur_s)} / {self.format_time(tot_s)}")

    def format_time(self, seconds):
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = BeerAnalysisApp()
    win.show()
    sys.exit(app.exec())