import sys
import os
import csv
import json
import re
from datetime import datetime, timedelta
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QFileDialog, QLabel, QSlider,
                             QDialog, QFormLayout, QDoubleSpinBox, QSpinBox, 
                             QComboBox, QDialogButtonBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMessageBox, QGridLayout)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QColor
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# IMPORTACIONES DE NUESTROS MODULOS
from gui_components import DetectionBar
from detector import AnalysisThread

def extract_tap_number(tap_id): #extraemos el numero que viene en el grifo para luego ejecutar la inferencia en base a las detecciones del propio surtidor y no en base a todas 
    
    import re
    if not tap_id:
        return None
    match = re.search(r'(\d+)', str(tap_id))
    if match:
        return int(match.group(1))
    return None

# --- 1. MODAL DE CONFIGURACIÓN ---
class SettingsModal(QDialog):
    def __init__(self, current_config, model_classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración Avanzada")
        self.setMinimumWidth(600)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        # (He eliminado el selector de duración aquí)

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
        
        return {
            'stride': self.stride_spin.value(),
            'iou': self.iou_spin.value(),
            # 'csv_duration': ... (ELIMINADO)
            'class_map': new_map
        }    
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

# --- 3. MODAL DE MÉTRICAS (KAPPA) --- #COMENTAR  
class MetricsModal(QDialog):
    def __init__(self, truth_bar, detection_bars, total_frames, stride, events_list, parent=None):
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

        # Obtener FPS del parent para calcular frames
        fps = parent.fps if parent and hasattr(parent, 'fps') else 15.0

        row = 0
        for name, bar in detection_bars.items():
            if len(bar.buffer) < total_frames: 
                continue
            
            # Extraer número de la clase (abierto_1 -> 1, abierto_2 -> 2, etc.)
            class_tap_number = extract_tap_number(name)
            
            # Construir y_true SOLO para este grifo
            y_true_raw = [0] * total_frames
            for evt in events_list:
                evt_tap_number = evt.get("tap_number")
                # Solo marcar si el grifo coincide
                if evt_tap_number == class_tap_number:
                    f_start = max(0, evt.get("f_start", 0))
                    f_end = min(total_frames - 1, evt.get("f_end", 0))
                    for f in range(f_start, f_end + 1):
                        if f < total_frames:
                            y_true_raw[f] = 1
            
            # Aplicar stride
            y_true = y_true_raw[::stride]
            y_pred = bar.buffer[:total_frames][::stride]

            # DEBUG (puedes quitar después)
            if "abierto" in name.lower():
                print(f"DEBUG [{name}]: tap_number={class_tap_number}, truth_frames={sum(y_true)}, pred_frames={sum(y_pred)}")

            try:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
            except:
                tn, fp, fn, tp = 0, 0, 0, 0

            kappa = cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
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

        layout.addWidget(QLabel(f"Nota: Métricas calculadas con Stride {stride}. Cada clase se compara con su grifo. en este porcentaje, puesto que es utilizado para calculo exacto de performance de los modelos no se añade granularidad ni tolerancia"))
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

# --- 4. MODAL DE LOG DE EVENTOS ---
class EventsLogModal(QDialog):
    def __init__(self, video_filename, detection_bars, fps, total_frames, current_events_in_range, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Desglose de Frames por Evento (Rango detectado)")
        self.setMinimumSize(1100, 600)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "ID Evento", "Clase Truth (CSV)", "Inicio Relativo", "Duración (Frames)", 
            "DETECCIONES POR CLASE (Frames Captados)", "Estado"
        ])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch) 
        layout.addWidget(self.table)

        row = 0
        for i, evt in enumerate(current_events_in_range):
            self.table.insertRow(row)
            json_class_name = evt["tap_id"]
            f_start = evt["f_start"]
            f_end = evt["f_end"]
            
            f_start = max(0, f_start)
            f_end = min(total_frames - 1, f_end)
            duration_frames = f_end - f_start
            
            if duration_frames <= 0: continue

            detections_summary = []
            for ai_name, ai_bar in detection_bars.items():
                fragment = ai_bar.buffer[f_start:f_end+1]
                frames_count = sum(1 for x in fragment if x > 0)
                if frames_count > 0:
                    detections_summary.append(f"{ai_name}: {frames_count}")

            if detections_summary:
                summary_str = " | ".join(detections_summary)
                status = "✅ CON DATOS"
                color = Qt.green
            else:
                summary_str = "--- (Silencio Total) ---"
                status = "❌ VACÍO"
                color = Qt.red

            self.table.setItem(row, 0, QTableWidgetItem(f"#{i+1}"))
            self.table.setItem(row, 1, QTableWidgetItem(json_class_name))
            
            rel_sec = f_start / fps if fps > 0 else 0
            self.table.setItem(row, 2, QTableWidgetItem(self.format_seconds(rel_sec)))
            self.table.setItem(row, 3, QTableWidgetItem(f"{duration_frames} f"))
            
            item_summary = QTableWidgetItem(summary_str)
            item_summary.setToolTip(summary_str)
            self.table.setItem(row, 4, item_summary)
            
            item_status = QTableWidgetItem(status)
            item_status.setForeground(color)
            item_status.setFont(self.get_bold_font())
            self.table.setItem(row, 5, item_status)
            row += 1

        layout.addWidget(QLabel(f"Total Eventos en rango: {row}."))
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

# --- 5. MODAL DE RESUMEN EJECUTIVO 
class SummaryModal(QDialog):
    def __init__(self, detected, missed, total_events, precision_pct, noise_seconds, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resumen Global de Calidad")
        self.setMinimumSize(600, 400)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout = QVBoxLayout(self)

        # --- SECCIÓN 1: EFICACIA (RECALL) ---
        lbl_eff = QLabel("1. EFICACIA (¿No se pierde nada?)")
        lbl_eff.setStyleSheet("font-size: 16px; font-weight: bold; color: #3498db; margin-top: 10px;")
        layout.addWidget(lbl_eff)

        grid = QGridLayout()
        
        lbl_det = QLabel("✅ Eventos Cazados:")
        lbl_det_val = QLabel(f"{detected}/{total_events}")
        lbl_det_val.setStyleSheet("color: #2ecc71; font-size: 24px; font-weight: bold;")
        grid.addWidget(lbl_det, 0, 0)
        grid.addWidget(lbl_det_val, 0, 1)

        lbl_miss = QLabel("❌ Eventos Perdidos:")
        lbl_miss_val = QLabel(f"{missed}")
        lbl_miss_val.setStyleSheet("color: #e74c3c; font-size: 24px; font-weight: bold;")
        grid.addWidget(lbl_miss, 1, 0)
        grid.addWidget(lbl_miss_val, 1, 1)
        
        layout.addLayout(grid)
        
        # --- SECCIÓN 2: CALIDAD (PRECISION) ---
        line = QLabel()
        line.setStyleSheet("border-top: 1px solid #555; margin: 10px 0;")
        layout.addWidget(line)
        
        lbl_qual = QLabel("2. FIABILIDAD (¿Inventa cosas?)")
        lbl_qual.setStyleSheet("font-size: 16px; font-weight: bold; color: #f1c40f; margin-top: 5px;")
        layout.addWidget(lbl_qual)
        
        grid2 = QGridLayout()
        
        # Porcentaje de Precisión
        lbl_prec = QLabel("🎯 Precisión de Detección:")
        lbl_prec_val = QLabel(f"{precision_pct:.1f}%")
        
        # Color dinámico para la nota
        color_prec = "#2ecc71" if precision_pct > 80 else "#f39c12" if precision_pct > 50 else "#e74c3c"
        lbl_prec_val.setStyleSheet(f"color: {color_prec}; font-size: 32px; font-weight: bold;")
        
        grid2.addWidget(lbl_prec, 0, 0)
        grid2.addWidget(lbl_prec_val, 0, 1)
        
        # Tiempo de Ruido
        lbl_noise = QLabel("🔊 Tiempo de 'Ruido' (Falso Positivo):")
        lbl_noise_val = QLabel(f"{noise_seconds:.1f} seg")
        lbl_noise_val.setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
        
        grid2.addWidget(lbl_noise, 1, 0)
        grid2.addWidget(lbl_noise_val, 1, 1)
        
        layout.addLayout(grid2)

        # Explicación
        lbl_note = QLabel("Nota: 'Precisión' indica qué porcentaje de lo que detecta la IA corresponde realmente a un evento del CSV.\nSi es baja, la IA está detectando grifos fantasma.")
        lbl_note.setStyleSheet("color: gray; font-size: 11px; margin-top: 15px;")
        lbl_note.setWordWrap(True)
        layout.addWidget(lbl_note)

        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

class VirtualVideoManager:
    def __init__(self):
        self.videos = [] # Lista de dicts: {path, start_global, end_global, total_frames, cap}
        self.total_global_frames = 0
        self.current_open_index = -1
        self.cap = None

    def add_video(self, path):
        # Abrimos momentáneamente para leer longitud
        temp_cap = cv2.VideoCapture(path)
        frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = temp_cap.get(cv2.CAP_PROP_FPS)
        temp_cap.release()

        start = self.total_global_frames
        end = start + frames - 1
        
        self.videos.append({
            'path': path,
            'start_global': start,
            'end_global': end,
            'frames': frames,
            'fps': fps
        })
        self.total_global_frames += frames

    def get_frame(self, global_frame_idx):
        # 1. Buscar en qué video cae este frame global
        target_video_idx = -1
        video_data = None
        
        for i, v in enumerate(self.videos):
            if v['start_global'] <= global_frame_idx <= v['end_global']:
                target_video_idx = i
                video_data = v
                break
        
        if target_video_idx == -1: return None, None # Fuera de rango

        # 2. Gestión inteligente de apertura de archivo (para no abrir/cerrar a lo loco)
        if self.current_open_index != target_video_idx:
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(video_data['path'])
            self.current_open_index = target_video_idx
            print(f"Cambio de cinta: Reproduciendo {os.path.basename(video_data['path'])}")

        # 3. Calcular frame local y buscar
        local_frame = global_frame_idx - video_data['start_global']
        
        # Optimización: Si pedimos el frame siguiente, no hacemos set (es lento), solo read
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if current_pos != local_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
            
        ret, frame = self.cap.read()
        return ret, frame
    
    def get_current_filename(self, global_frame_idx):
        for v in self.videos:
            if v['start_global'] <= global_frame_idx <= v['end_global']:
                return os.path.basename(v['path'])
        return "Desconocido"

# --- 1. MODAL DE CONFIGURACIÓN (CORREGIDO) ---
class ConfusionMatrixModal(QDialog):
    def __init__(self, y_true, y_pred, class_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Matriz de Confusión: {class_name}")
        self.setMinimumSize(500, 500)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
               
        layout = QVBoxLayout(self)
        
        # Calcular métricas usando sklearn
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0]) # 1: Abierto, 0: Cerrado
        tp, fn, fp, tn = cm.ravel()
        
        # --- CORRECCIÓN MATEMÁTICA ---
        # 1. Calculamos los totales POR FILA (Realidad)
        total_real_abierto = tp + fn
        total_real_cerrado = fp + tn
        
        # 2. Prevenimos división por cero (si el video o CSV está vacío en esa categoría)
        denom_abierto = total_real_abierto if total_real_abierto > 0 else 1
        denom_cerrado = total_real_cerrado if total_real_cerrado > 0 else 1
        
        # 3. Calculamos porcentajes relativos a la FILA
        pct_tp = (tp / denom_abierto) * 100
        pct_fn = (fn / denom_abierto) * 100
        pct_fp = (fp / denom_cerrado) * 100
        pct_tn = (tn / denom_cerrado) * 100
        # -----------------------------

        # Título
        title = QLabel(f"Análisis de Frames para: {class_name}")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #f1c40f; margin-bottom: 10px;")
        layout.addWidget(title)

        # Grid de la Matriz
        grid = QGridLayout()
        grid.setSpacing(10)

        # Labels de cabecera
        grid.addWidget(QLabel(""), 0, 0)
        grid.addWidget(self._header_label("PREDICCIÓN IA\n(Abierto)"), 0, 1)
        grid.addWidget(self._header_label("PREDICCIÓN IA\n(Cerrado)"), 0, 2)

        # Fila 1: Realidad Abierto (Usamos pct_tp y pct_fn)
        # Añadí el total de la fila en el texto para mayor claridad
        grid.addWidget(self._header_label(f"REALIDAD\n(CSV Abierto)\nN={total_real_abierto}"), 1, 0)
        grid.addWidget(self._cell_box(f"TP (Acierto)\n{tp}", f"{pct_tp:.1f}%", "#2ecc71"), 1, 1)
        grid.addWidget(self._cell_box(f"FN (Olvido)\n{fn}", f"{pct_fn:.1f}%", "#e74c3c"), 1, 2)

        # Fila 2: Realidad Cerrado (Usamos pct_fp y pct_tn)
        grid.addWidget(self._header_label(f"REALIDAD\n(CSV Cerrado)\nN={total_real_cerrado}"), 2, 0)
        grid.addWidget(self._cell_box(f"FP (Falsa Alarma)\n{fp}", f"{pct_fp:.1f}%", "#e67e22"), 2, 1)
        grid.addWidget(self._cell_box(f"TN (Correcto)\n{tn}", f"{pct_tn:.1f}%", "#34495e"), 2, 2)

        layout.addLayout(grid)

        # Métricas resumen al pie
        total_global = tp + fn + fp + tn
        precision = (tp/(tp+fp)*100) if (tp+fp)>0 else 0
        recall = (tp/(tp+fn)*100) if (tp+fn)>0 else 0
        
        metrics_text = (
            f"<b>Precisión:</b> {precision:.2f}% (Fiabilidad de lo detectado)<br>"
            f"<b>Recall:</b> {recall:.2f}% (Capacidad de no perder eventos)<br>"
            f"<b>Total Frames Analizados:</b> {total_global}"
            f"<br><i>Nota: Porcentajes calculados respecto a la realidad de cada clase (fila), no al total global. con una tolerancia de 2.5 (medimos precision estricta a nivel de tolerancia)</i>"
        )
        lbl_metrics = QLabel(metrics_text)
        lbl_metrics.setStyleSheet("background: #2b2b2b; padding: 15px; border-radius: 5px; margin-top: 10px;")
        layout.addWidget(lbl_metrics)

        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def _header_label(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-weight: bold; color: #95a5a6; font-size: 11px;")
        return lbl

    def _cell_box(self, top_text, pct_text, color):
        widget = QWidget()
        widget.setStyleSheet(f"background-color: {color}; border-radius: 10px; min-height: 100px;")
        l = QVBoxLayout(widget)
        
        t1 = QLabel(top_text)
        t1.setAlignment(Qt.AlignCenter)
        t1.setStyleSheet("color: white; font-weight: bold; font-size: 14px; background: transparent;")
        
        t2 = QLabel(pct_text)
        t2.setAlignment(Qt.AlignCenter)
        t2.setStyleSheet("color: rgba(255,255,255,0.8); font-size: 18px; font-weight: bold; background: transparent;")
        
        l.addWidget(t1)
        l.addWidget(t2)
        return widget

# --- APLICACIÓN PRINCIPAL ---
class BeerAnalysisApp(QMainWindow):
    def __init__(self): 
        super().__init__()  # Inicializamos app 
        self.setWindowTitle("Validador de Bebidas IA - Gamb00za 2026")
        self.resize(1200, 900)
        self.setStyleSheet("QMainWindow { background-color: #121212; } QPushButton { padding: 8px; color: white; background-color: #333; }")
        
        self.virtual_manager = VirtualVideoManager() # Funcion que estamos implementando para que se puedan ejecutar múltiples vídeos como uno solo
        self.is_global_mode = False                  # Indica si estamos en modo vídeo único o global (múltiples vídeos)
        
        self.video_path = None # Igualamos a None para evitar errores antes de darle el verdadero path tanto de modelo como de video
        self.model_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 15.0
        self.model_classes = []
        self.detection_bars = {} 
        self.ai_config = {'conf': 0.40, 'stride': 1, 'class_map': {}}  # Configuración IA por defecto
        self.last_csv_path = None # <--- Variable nueva para recordar qué CSV recargar
        self.ai_config = {'conf': 0.40, 'stride': 1, 'class_map': {}} 
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.analysis_thread = None
        self.video_queue = []
        self.video_root_folder = ""
        self.is_batch_running = False
        
        self.all_csv_events = [] 
        self.current_video_events = []

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Fila Archivos
        files_layout = QHBoxLayout() # fila de botones de carga
        self.btn_folder = QPushButton("1. 📁 Carpeta Videos")
        self.btn_folder.clicked.connect(self.select_folder)
        self.btn_vid = QPushButton("2. 📂 Cargar 1 Video")
        self.btn_vid.clicked.connect(self.load_video)
        self.btn_mod = QPushButton("3. 🧠 Cargar Modelo")
        self.btn_mod.clicked.connect(self.load_model)
        self.btn_csv = QPushButton("4. 📄 Cargar CSV (Truth)")
        self.btn_csv.clicked.connect(self.load_csv)
        
        for b in [self.btn_folder, self.btn_vid, self.btn_mod, self.btn_csv]: files_layout.addWidget(b)
        layout.addLayout(files_layout)

        # Fila Ejecución
        action_layout = QHBoxLayout()
        self.btn_settings = QPushButton("⚙ Ajustar IA")
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_run = QPushButton("▶ EJECUTAR VIDEO ÚNICO")
        self.btn_run.setStyleSheet("background-color: #2e7d32; font-weight: bold;")
        self.btn_run.clicked.connect(self.start_analysis)
        self.btn_run_batch = QPushButton("⏩ EJECUTAR LOTE (CSV)")
        self.btn_run_batch.setStyleSheet("background-color: #d35400; font-weight: bold;")
        self.btn_run_batch.setEnabled(False) 
        self.btn_run_batch.clicked.connect(self.start_batch_processing)
        action_layout.addWidget(self.btn_settings)
        action_layout.addWidget(self.btn_run)
        action_layout.addWidget(self.btn_run_batch)
        layout.addLayout(action_layout)
        
        # Fila Resultados
        # Fila Resultados
        results_layout = QHBoxLayout()
        # Cambiamos el texto del botón para que refleje la nueva función técnica
        self.btn_results = QPushButton("🎯 Matriz de Confusión") 
        self.btn_results.setStyleSheet("background-color: #27ae60; font-weight: bold; color: white;")
        self.btn_results.clicked.connect(self.show_results)

        self.btn_metrics = QPushButton("📈 Ver Métricas (Kappa)")
        self.btn_metrics.setStyleSheet("background-color: #8e44ad; font-weight: bold;")
        self.btn_metrics.clicked.connect(self.show_metrics)

        self.btn_log = QPushButton("📋 Log Eventos")
        self.btn_log.setStyleSheet("background-color: #7f8c8d; font-weight: bold;")
        self.btn_log.clicked.connect(self.show_event_log)
        
        # --- Boton de resumen
        self.btn_summary = QPushButton("🏆 Ver Resumen")
        self.btn_summary.setStyleSheet("background-color: #f1c40f; color: black; font-weight: bold;")
        self.btn_summary.clicked.connect(self.show_summary)
        
        results_layout.addWidget(self.btn_results)
        results_layout.addWidget(self.btn_metrics)
        results_layout.addWidget(self.btn_log)
        results_layout.addWidget(self.btn_summary)
        layout.addLayout(results_layout)

        # Monitor
        self.video_display = QLabel("Seleccione Carpeta, Modelo y CSV")
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

    
    def show_results(self):

        
        #   MATRIZ DE CONFUSIÓN GLOBAL
        #   ==========================
        #   Mide el rendimiento del sistema COMPLETO (todos los grifos combinados).
            
        #    Pregunta que responde: "¿La IA detectó actividad de cerveza cuando 
        #    realmente la había, sin importar en qué grifo específico?"
            
        #    - NO distingue entre grifos individuales
        #   - Útil para evaluar el sistema en conjunto
        #   - Para métricas por grifo específico, usar "Informe Técnico (Kappa)"
        
        # Validaciones básicas
        if not self.current_video_events or self.total_frames == 0:
            QMessageBox.warning(self, "Aviso", "No hay datos suficientes (CSV o Video vacío).")
            return
        
        # 1. Definir qué clases son 'Grifos' para unificar detecciones
        target_keywords = ["grifo", "abierto_1", "abierto_2", "abierto_3", "abierto_4", "abierto_5"] # definimos las keywords que identifican grifos (para parnasillo u otros modelos)
        valid_bars = [bar for name, bar in self.detection_bars.items() 
                      if any(k in name.lower() for k in target_keywords)] # filtramos solo las clases que coinciden con las keywords
        
        if not valid_bars:
            QMessageBox.warning(self, "Error", "No se encontraron clases de tipo 'grifo'.")
            return

        
        # Array de PREDICCIÓN (IA)
        # Si CUALQUIERA de los grifos detecta algo en frame i -> pred[i] = 1
        y_pred_full = [0] * self.total_frames # Inicializamos todo a 0
        for bar in valid_bars:  # Recorremos solo las barras válidas
            for i, val in enumerate(bar.buffer):
                if val > 0: y_pred_full[i] = 1 # Marcamos como 1 si cualquiera detecta

        # B) Array de REALIDAD (CSV)
        # Aplicamos 2.5s de margen para ser justos con el lag del visor GRANULARIDAD
        tolerance_frames = int(2.5 * self.fps) 
        y_true_full = [0] * self.total_frames
        
        for evt in self.current_video_events:
            f_start = max(0, evt["f_start"] - tolerance_frames)
            f_end = min(self.total_frames - 1, evt["f_end"] + tolerance_frames)
            # Rellenamos de 1s la zona del evento + tolerancia
            for i in range(f_start, f_end + 1):
                y_true_full[i] = 1

        # 3. APLICAR EL STRIDE (Muestreo)
        # Esta es la parte CLAVE. Si stride=5, nos quedamos con frames 0, 5, 10...
        current_stride = self.ai_config.get('stride', 1)
        
        # Slicing de listas en Python: lista[inicio:fin:paso]
        y_true_sampled = y_true_full[::current_stride]
        y_pred_sampled = y_pred_full[::current_stride]

        # 4. Lanzar Modal
        modal = ConfusionMatrixModal(y_true_sampled, y_pred_sampled, current_stride, self)
        modal.exec()

    # --- HELPERS DE FECHA Y MATCHING ---
    def extract_datetime_from_filename(self, filename):
        try:
            name = os.path.splitext(filename)[0]
            parts = name.split('_')
            if not parts: return None
            for p in reversed(parts):
                if len(p) == 14 and p.isdigit():
                    return datetime.strptime(p, "%Y%m%d%H%M%S")
        except: pass
        return None

    def extract_date_from_text(self, text):
        if not text: return None
        try:
            match14 = re.search(r'(\d{14})', text)
            if match14: return datetime.strptime(match14.group(1), "%Y%m%d%H%M%S")
            match8 = re.search(r'(\d{8})', text)
            if match8: return datetime.strptime(match8.group(1), "%Y%m%d")
        except: pass
        return None

    

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV Truth", "", "CSV (*.csv)")
        if path:
            self.load_csv_file(path)

    def load_csv_file(self, path):
        try:
            self.last_csv_path = path 
            self.all_csv_events = [] 
            count_entries = 0
            count_skipped = 0 
            
            with open(path, 'r', encoding='utf-8-sig') as f:
                sample = f.read(2048)
                f.seek(0)
                delimiter = ';' if ';' in sample else ','
                reader = csv.DictReader(f, delimiter=delimiter)
                
                # --- 1. DETECCIÓN INTELIGENTE DE COLUMNA CERVEZA ---
                headers = reader.fieldnames if reader.fieldnames else []
                beer_col = None
                for h in headers:
                    h_clean = h.lower().strip()
                    if "cerveza" in h_clean or "beer" in h_clean:
                        beer_col = h
                        break
                
                if not beer_col:
                    QMessageBox.warning(self, "Atención", 
                        f"No he encontrado la columna 'Cerveza'.\nColumnas: {headers}")
                    return

                for row in reader:
                    # --- 2. FILTRO FLEXIBLE ---
                    raw_val = row.get(beer_col, "0")
                    val_str = str(raw_val).strip().lower()
                    if val_str not in ['1', 'true', 'si', 'yes', 's', 'y']:
                        count_skipped += 1
                        continue 

                    tap_id = row.get("Grifo") or row.get("grifo") or "Unknown"
                    url_video = row.get("URL Video", "")
                    
                    dt_from_url = self.extract_date_from_text(url_video)
                    if not dt_from_url:
                        v_path = row.get("Video Path", "")
                        if v_path: dt_from_url = self.extract_datetime_from_filename(os.path.basename(v_path))
                    if not dt_from_url: dt_from_url = datetime.now() 

                    # --- 3. VOLVEMOS A LEER HORA INICIO Y HORA FIN ---
                    start_str = row.get("Hora Inicio", "00:00:00")
                    end_str = row.get("Hora Fin", "00:00:00") # <--- Recuperado
                    
                    try:
                        t_start = datetime.strptime(start_str.strip(), "%H:%M:%S").time()
                        t_end = datetime.strptime(end_str.strip(), "%H:%M:%S").time() # <--- Recuperado
                        
                        base_date = dt_from_url.date() if isinstance(dt_from_url, datetime) else dt_from_url
                        
                        start_dt = datetime.combine(base_date, t_start)
                        end_dt = datetime.combine(base_date, t_end)
                        
                        # Si la hora fin es menor que la inicio, asumimos que ha pasado la medianoche (día siguiente)
                        if end_dt < start_dt: 
                            end_dt += timedelta(days=1)
                        
                    except Exception as e: 
                        continue

                    self.all_csv_events.append({
                        "tap_id": tap_id,
                        "tap_number": extract_tap_number(tap_id),
                        "start_dt": start_dt,
                        "end_dt": end_dt
                    })
                    count_entries += 1

            self.btn_csv.setText(f"📄 CSV (Real) - {count_entries} Cervezas")
            
            if self.is_global_mode:
                self.load_global_truth()
            elif self.video_path: 
                self.load_truth_for_current_video(os.path.basename(self.video_path))
            
            self.check_batch_ready()

        except Exception as e:
            print(f"❌ Error CSV: {e}")
            QMessageBox.warning(self, "Error CSV", f"No se pudo procesar:\n{e}")



    def load_truth_for_current_video(self, filename):
        video_start_dt = self.extract_datetime_from_filename(filename)
        if not video_start_dt: return

        duration_sec = self.total_frames / self.fps if self.fps > 0 else 0
        video_end_dt = video_start_dt + timedelta(seconds=duration_sec)
        
        self.bar_truth.init_buffer(self.total_frames)
        self.current_video_events = [] 
        real_fps = self.fps if self.fps > 0 else 15.0
        
        for event in self.all_csv_events:
            ev_start = event["start_dt"]
            ev_end = event["end_dt"]
            if ev_start < video_end_dt and ev_end > video_start_dt:
                rel_start_sec = (ev_start - video_start_dt).total_seconds()
                rel_end_sec = (ev_end - video_start_dt).total_seconds()
                f_start = int(rel_start_sec * real_fps)
                f_end = int(rel_end_sec * real_fps)
                
                event_copy = event.copy()
                event_copy["f_start"] = f_start
                event_copy["f_end"] = f_end
                self.current_video_events.append(event_copy)

                draw_start = max(0, f_start)
                draw_end = min(self.total_frames - 1, f_end)
                if draw_end > draw_start:
                    for f in range(draw_start, draw_end + 1):
                        self.bar_truth.mark_detection(f, True)
        self.bar_truth.update()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta con Videos")
        if folder:
            # 1. Guardar la ruta
            self.video_root_folder = folder
            
            # 2. Contar videos inmediatamente para confirmar al usuario
            videos_found = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi'))]
            count = len(videos_found)
            
            # 3. Actualizar texto del botón
            self.btn_folder.setText(f"📁 .../{os.path.basename(folder)} ({count} videos)")
            
            # 4. Avisar si está vacía
            if count == 0:
                QMessageBox.warning(self, "Carpeta Vacía", "No he encontrado videos .mp4 o .avi en esta carpeta.")
                self.btn_run_batch.setEnabled(False) # Asegurar que no se pueda ejecutar
            else:
                # 5. Comprobar si ya podemos activar el botón de Lote
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
        if self.all_csv_events: self.load_truth_for_current_video(os.path.basename(path))

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

    def check_batch_ready(self):
        has_folder = bool(self.video_root_folder)
        has_csv = bool(self.all_csv_events)
        
        if has_folder and has_csv:
            self.btn_run_batch.setEnabled(True)
            self.btn_run_batch.setText(f"⏩ EJECUTAR LOTE ({len(self.all_csv_events)} eventos)")
            self.btn_run_batch.setStyleSheet("background-color: #d35400; font-weight: bold; color: white;")
        else:
            self.btn_run_batch.setEnabled(False)
            # Feedback visual de qué falta
            missing = []
            if not has_folder: missing.append("Falta Carpeta")
            if not has_csv: missing.append("Falta CSV")
            self.btn_run_batch.setText(f"⏩ Esperando... ({', '.join(missing)})")
            self.btn_run_batch.setStyleSheet("background-color: #555; color: #aaa;")

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
            # Ya no necesitamos recargar el CSV al cambiar ajustes
            # porque la duración ahora viene fija en el archivo, no en la config.
    
    def run_global_analysis_step(self, video_index):
        """ Ejecuta el análisis del video N de la lista virtual """
        if video_index >= len(self.virtual_manager.videos):
            QMessageBox.information(self, "Fin", "Análisis de Lote Global Completado.")
            self.show_summary() # Resumen de TODO el lote
            return

        video_data = self.virtual_manager.videos[video_index]
        path = video_data['path']
        global_offset = video_data['start_global']
        
        # Configuramos el título para saber progreso
        self.setWindowTitle(f"Analizando {video_index + 1}/{len(self.virtual_manager.videos)}: {os.path.basename(path)}")
        
        # Reutilizamos tu AnalysisThread existente
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.stop()
            self.analysis_thread.wait()
            
        # IMPORTANTE: Pasamos '0' como start_frame local, 
        # pero necesitamos pasarle el offset global para que sepa dónde pintar
        self.analysis_thread = AnalysisThread(path, self.model_path, self.ai_config, 0)
        
        # Conectamos las señales
        self.analysis_thread.frame_ready.connect(self.display_analysis_frame)
        
        # TRUCO: Interceptamos la señal de detección para sumar el offset
        self.analysis_thread.detection_event.connect(lambda f, d: self.update_global_bars(f, d, global_offset))
        
        # Al terminar este video, lanzamos el siguiente
        self.analysis_thread.finished.connect(lambda: self.run_global_analysis_step(video_index + 1))
        
        self.analysis_thread.start()

    def update_global_bars(self, local_frame, detections_dict, global_offset):
        # Convertimos frame local a global
        global_frame = global_offset + local_frame
        
        # Pintamos en las barras globales
        for class_name, found in detections_dict.items():
            if class_name in self.detection_bars:
                bar = self.detection_bars[class_name]
                bar.mark_detection(global_frame, found)
                bar.set_current_frame(global_frame)
        
        self.bar_truth.set_current_frame(global_frame)
        self.timeline_bar.setValue(global_frame)
        # Opcional: No actualizar video_position cada frame para ganar velocidad
        # self.set_video_position(global_frame)

    def start_batch_processing(self):
        if not self.model_path or not self.video_root_folder: return

        # 1. Reiniciar Gestor Virtual
        self.virtual_manager = VirtualVideoManager()
        self.is_global_mode = True
        
        # 2. Escanear todos los videos y añadirlos al timeline virtual
        video_files = sorted([f for f in os.listdir(self.video_root_folder) if f.lower().endswith(('.mp4', '.avi'))])
        
        if not video_files:
            QMessageBox.warning(self, "Error", "Carpeta vacía")
            return

        progress_dialog = QMessageBox(self)
        progress_dialog.setText("Escaneando duración de videos... (Esto no tarda mucho)")
        progress_dialog.setStandardButtons(QMessageBox.NoButton)
        progress_dialog.show()
        QApplication.processEvents()

        for f in video_files:
            full_path = os.path.join(self.video_root_folder, f)
            self.virtual_manager.add_video(full_path)

        progress_dialog.close()
        progress_dialog.deleteLater()  # <--- Asegura que se destruya el objeto
        QApplication.processEvents()   # <--- ¡IMPORTANTE! Fuerza a la pantalla a actualizarse YA

        # 3. Configurar UI Global
        self.total_frames = self.virtual_manager.total_global_frames
        self.timeline_bar.setRange(0, self.total_frames - 1)
        self.lbl_time.setText(f"LOTE COMPLETO: {len(video_files)} Videos")

        # 4. Inicializar Barras Gigantes
        for bar in self.detection_bars.values(): 
            bar.init_buffer(self.total_frames)
        self.bar_truth.init_buffer(self.total_frames)

        # 5. Cargar Truth Global (Mapear CSV a la línea de tiempo gigante)
        self.load_global_truth()

        # 6. Iniciar Análisis (Aquí lanzamos el thread iterando sobre lo global)
        # Nota: Para simplificar, aquí usaremos una recursividad modificada o un thread que sepa cambiar de video.
        # Por ahora, cargamos el primer frame visualmente:
        self.set_video_position(0)
        
        # Lanzamos el proceso de análisis continuo
        self.run_global_analysis_step(0)

    def load_global_truth(self):
        """Mapea los eventos del CSV a la posición global exacta en la cadena de videos"""
        self.current_video_events = [] # Usaremos esto para el reporte global
        
        # Iteramos sobre cada 'segmento' de video virtual
        for v_data in self.virtual_manager.videos:
            filename = os.path.basename(v_data['path'])
            v_start_dt = self.extract_datetime_from_filename(filename)
            if not v_start_dt: continue
            
            # Buscamos eventos que coincidan con ESTE video específico
            # Lógica similar a load_truth_for_current_video pero ajustando el offset
            video_end_dt = v_start_dt + timedelta(seconds=v_data['frames']/v_data['fps'])
            
            for event in self.all_csv_events:
                if event["start_dt"] >= v_start_dt and event["end_dt"] <= video_end_dt:
                    # Calcular frame relativo al video
                    rel_start_sec = (event["start_dt"] - v_start_dt).total_seconds()
                    rel_end_sec = (event["end_dt"] - v_start_dt).total_seconds()
                    
                    local_f_start = int(rel_start_sec * v_data['fps'])
                    local_f_end = int(rel_end_sec * v_data['fps'])
                    
                    # CONVERTIR A GLOBAL
                    global_f_start = v_data['start_global'] + local_f_start
                    global_f_end = v_data['start_global'] + local_f_end
                    
                    # Guardar y Pintar
                    self.current_video_events.append({
                        **event, 
                        "f_start": global_f_start, 
                        "f_end": global_f_end
                    })
                    
                    for f in range(global_f_start, global_f_end + 1):
                        if f < self.total_frames:
                            self.bar_truth.mark_detection(f, True)
        
        self.bar_truth.update()

    def process_next_in_queue(self):
        if not self.video_queue:
            self.is_batch_running = False
            QMessageBox.information(self, "Fin", "Lote procesado.")
            return
        video_filename = self.video_queue.pop(0)
        full_path = os.path.join(self.video_root_folder, video_filename)
        self.load_video_from_path(full_path)
        self.start_analysis(auto_batch=True)

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
            
            # CONECTAMOS EL FINAL DEL ANÁLISIS AL RESUMEN
            if auto_batch:
                self.analysis_thread.finished.connect(self.process_next_in_queue)
            else:
                self.analysis_thread.finished.connect(self.on_analysis_finished) 
                
            self.analysis_thread.start()

    def on_analysis_finished(self):
        self.btn_play.setText("▶ Play") 
        self.show_summary() # Lanzar popup

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
        if self.is_global_mode:
            # --- MODO LOTE GLOBAL ---
            ret, img_cv = self.virtual_manager.get_frame(pos)
            if ret:
                rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
                self.display_analysis_frame(img)
                
                # Actualizar título con el video actual
                curr_name = self.virtual_manager.get_current_filename(pos)
                self.setWindowTitle(f"LOTE GLOBAL - Viendo: {curr_name} (Frame Global: {pos})")
        else:
            # --- MODO VIDEO ÚNICO (Tu código original) ---
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = self.cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
                    self.display_analysis_frame(img)

        # Común para ambos
        for bar in self.detection_bars.values(): bar.set_current_frame(pos)
        self.bar_truth.set_current_frame(pos)
        self.update_time_label(pos)

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
        else:
            self.timer.stop()
            self.btn_play.setText("▶ Play")

    def show_summary(self):
        if not self.current_video_events:
            QMessageBox.warning(self, "Info", "Sin datos de eventos para este video.")
            return
        
        # --- 1. CONFIGURACIÓN DE GRANULARIDAD ---
        # Tolerancia: 2.5 segundos a cada lado es un estándar razonable
        tolerance_seconds = 2.5
        tolerance_frames = int(tolerance_seconds * self.fps)
        
        target_keywords = ["grifo", "tap", "handle", "abierto_1", "abierto_2", "abierto_3", "abierto_4", "abierto_5"] 
        valid_ai_bars = [bar for name, bar in self.detection_bars.items() 
                        if any(k in name.lower() for k in target_keywords)]

        if not valid_ai_bars:
            QMessageBox.warning(self, "Aviso", "No se detectaron clases de tipo 'grifo'.")
            return

        # --- 2. CONSTRUCCIÓN DE MÁSCARAS ---
        truth_mask = [False] * self.total_frames
        # Esta es la clave: la máscara donde 'todo vale' (dentro del rango + tolerancia)
        truth_mask_extended = [False] * self.total_frames

        for evt in self.current_video_events:
            f_start = max(0, evt.get("f_start", 0))
            f_end = min(self.total_frames - 1, evt.get("f_end", 0))
            
            if f_end <= f_start: continue

            # Llenamos la verdad absoluta
            for i in range(f_start, f_end + 1):
                truth_mask[i] = True
                
            # Llenamos la zona de tolerancia (Granularidad)
            t_start = max(0, f_start - tolerance_frames)
            t_end = min(self.total_frames - 1, f_end + tolerance_frames)
            for i in range(t_start, t_end + 1):
                truth_mask_extended[i] = True

        # --- 3. CÁLCULO DE EFICACIA (RECALL) ---
        detected_events = 0
        missed_events = 0
        for evt in self.current_video_events:
            f_start, f_end = evt["f_start"], evt["f_end"]
            # Aquí usamos la máscara original para ver si 'cazamos' el evento
            any_det = any(sum(bar.buffer[max(0,f_start):min(self.total_frames,f_end+1)]) > 0 
                        for bar in valid_ai_bars)
            if any_det: detected_events += 1
            else: missed_events += 1

        # --- 4. CÁLCULO DE PRECISIÓN (FIABILIDAD CON TOLERANCIA) ---
        ai_true_positive_frames = 0
        ai_false_positive_frames = 0
        
        # Unificamos actividad de todos los grifos en una sola línea de tiempo
        ai_activity = [False] * self.total_frames
        for bar in valid_ai_bars:
            for i, val in enumerate(bar.buffer):
                if val > 0: ai_activity[i] = True
        
        for i in range(self.total_frames):
            if ai_activity[i]: 
                # Si la IA detecta algo, miramos si está en la zona EXTENDIDA
                if truth_mask_extended[i]:
                    ai_true_positive_frames += 1
                else:
                    ai_false_positive_frames += 1
        
        total_ai_frames = ai_true_positive_frames + ai_false_positive_frames
        precision_pct = (ai_true_positive_frames / total_ai_frames * 100) if total_ai_frames > 0 else 0.0
        noise_seconds = ai_false_positive_frames / self.fps if self.fps > 0 else 0

        modal = SummaryModal(detected_events, missed_events, len(self.current_video_events), 
                            precision_pct, noise_seconds, self)
        modal.exec()
    

    def show_metrics(self):
        if self.total_frames > 0:
            stride = self.ai_config.get('stride', 1)
            modal = MetricsModal(self.bar_truth, self.detection_bars, self.total_frames, stride, self.current_video_events, self)
            modal.exec()

    def show_event_log(self):
        if not self.video_path or not self.all_csv_events:
            QMessageBox.warning(self, "Faltan datos", "Carga un video y el CSV primero.")
            return
        modal = EventsLogModal(os.path.basename(self.video_path), self.detection_bars, self.fps, self.total_frames, self.current_video_events, self)
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