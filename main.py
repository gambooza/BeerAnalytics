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

class ConfusionMatrixModal(QDialog):
    def __init__(self, y_true, y_pred, class_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Matriz de Confusión: {class_name}")
        self.setMinimumSize(500, 500)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
        
        layout = QVBoxLayout(self)
        
        # Calcular métricas usando sklearn
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0]) # 1: Abierto, 0: Cerrado
        # Extraer valores (notar que labels=[1,0] cambia el orden estándar a TP, FN, FP, TN)
        tp, fn, fp, tn = cm.ravel()
        total = tp + tn + fp + fn
        
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

        # Fila 1: Realidad Abierto
        grid.addWidget(self._header_label("REALIDAD\n(CSV Abierto)"), 1, 0)
        grid.addWidget(self._cell_box(f"TP (Acierto)\n{tp}", f"{(tp/total*100):.1f}%", "#2ecc71"), 1, 1)
        grid.addWidget(self._cell_box(f"FN (Olvido)\n{fn}", f"{(fn/total*100):.1f}%", "#e74c3c"), 1, 2)

        # Fila 2: Realidad Cerrado
        grid.addWidget(self._header_label("REALIDAD\n(CSV Cerrado)"), 2, 0)
        grid.addWidget(self._cell_box(f"FP (Fantasma)\n{fp}", f"{(fp/total*100):.1f}%", "#e67e22"), 2, 1)
        grid.addWidget(self._cell_box(f"TN (Silencio)\n{tn}", f"{(tn/total*100):.1f}%", "#34495e"), 2, 2)

        layout.addLayout(grid)

        # Métricas resumen al pie
        metrics_text = (
            f"<b>Precisión:</b> {(tp/(tp+fp)*100 if (tp+fp)>0 else 0):.2f}% (Fiabilidad)<br>"
            f"<b>Recall:</b> {(tp/(tp+fn)*100 if (tp+fn)>0 else 0):.2f}% (Eficacia)<br>"
            f"<b>Total Frames Analizados:</b> {total}"
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

# --- 5. MODAL DE RESUMEN EJECUTIVO ---
# --- 5. MODAL DE RESUMEN EJECUTIVO (COMPLETO) ---
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
        
        self.all_csv_events = [] 
        self.current_video_events = []

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
        
        # --- NUEVO BOTÓN MANUAL DE RESUMEN ---
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

    # Pega esto dentro de class BeerAnalysisApp, a la altura de def init_ui o def load_video
    def show_results(self):
        if not self.current_video_events or self.total_frames == 0:
            QMessageBox.warning(self, "Aviso", "No hay datos suficientes para generar la matriz.")
            return

        # 1. Crear la Verdad (Truth Mask) con Tolerancia de 1.5 seg
        tolerance_frames = int(1.5 * self.fps) 
        y_true_extended = [0] * self.total_frames
        
        for evt in self.current_video_events:
            f_start = max(0, evt["f_start"] - tolerance_frames)
            f_end = min(self.total_frames - 1, evt["f_end"] + tolerance_frames)
            for i in range(f_start, f_end + 1):
                y_true_extended[i] = 1

        # 2. Elegir qué clases analizar (Grifos)
        target_keywords = ["grifo", "tap", "handle", "abierto_1"]
        valid_bars = [bar for name, bar in self.detection_bars.items() 
                      if any(k in name.lower() for k in target_keywords)]

        if not valid_bars:
            QMessageBox.warning(self, "Error", "No hay clases de 'grifo' detectadas (revisa nombres de clases).")
            return

        # 3. Unificamos las detecciones de la IA
        y_pred = [0] * self.total_frames
        for bar in valid_bars:
            for i, val in enumerate(bar.buffer):
                if val > 0: y_pred[i] = 1

        # 4. Lanzar el Modal de Matriz
        # Usamos stride=1 para máxima precisión en el análisis final
        modal = ConfusionMatrixModal(y_true_extended, y_pred, "Grifos (Unificado)", self)
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

    def find_best_match(self, json_name, ai_names):
        """Busca la clase de IA que mejor encaja con el nombre del JSON"""
        if json_name in ai_names: return json_name, "Exacto"
        json_lower = json_name.lower()
        for name in ai_names:
            if name.lower() == json_lower: return name, "Case-Insensitive"
        candidates = []
        for name in ai_names:
            if json_name in name or name in json_name: candidates.append(name)
        if candidates:
            candidates.sort(key=lambda x: abs(len(x) - len(json_name)))
            return candidates[0], "Aproximado"
        return None, "Ninguno"

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV Truth", "", "CSV (*.csv)")
        if path:
            try:
                self.all_csv_events = [] 
                count_entries = 0
                with open(path, 'r', encoding='utf-8-sig') as f:
                    sample = f.read(2048)
                    f.seek(0)
                    delimiter = ';' if ';' in sample else ','
                    reader = csv.DictReader(f, delimiter=delimiter)
                    for row in reader:
                        tap_id = row.get("Grifo") or row.get("grifo") or "Unknown"
                        url_video = row.get("URL Video", "")
                        
                        dt_from_url = self.extract_date_from_text(url_video)
                        if not dt_from_url:
                            v_path = row.get("Video Path", "")
                            if v_path: dt_from_url = self.extract_datetime_from_filename(os.path.basename(v_path))
                        if not dt_from_url: dt_from_url = datetime.now() 

                        start_str = row.get("Hora Inicio", "00:00:00")
                        end_str = row.get("Hora Fin", "00:00:00")
                        
                        try:
                            t_start = datetime.strptime(start_str.strip(), "%H:%M:%S").time()
                            t_end = datetime.strptime(end_str.strip(), "%H:%M:%S").time()
                            base_date = dt_from_url.date() if isinstance(dt_from_url, datetime) else dt_from_url
                            start_dt = datetime.combine(base_date, t_start)
                            end_dt = datetime.combine(base_date, t_end)
                            
                            if end_dt < start_dt: end_dt += timedelta(days=1)
                        except: continue

                        self.all_csv_events.append({
                            "tap_id": tap_id,
                            "start_dt": start_dt,
                            "end_dt": end_dt
                        })
                        count_entries += 1

                self.btn_csv.setText(f"📄 CSV Universal ({count_entries})")
                if self.video_path: self.load_truth_for_current_video(os.path.basename(self.video_path))
                self.check_batch_ready()
            except Exception as e:
                print(f"❌ Error CSV: {e}")

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
        if self.video_root_folder and self.all_csv_events:
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
        self.video_queue = [f for f in os.listdir(self.video_root_folder) if f.lower().endswith(('.mp4', '.avi'))]
        self.is_batch_running = True
        self.process_next_in_queue()

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
        else:
            self.timer.stop()
            self.btn_play.setText("▶ Play")

    def show_summary(self):
        if not self.current_video_events:
            QMessageBox.warning(self, "Info", "Sin datos de eventos para este video.")
            return
        
        # --- 1. CONFIGURACIÓN DE GRANULARIDAD ---
        # Tolerancia: 1.5 segundos a cada lado es un estándar razonable
        tolerance_seconds = 1.5 
        tolerance_frames = int(tolerance_seconds * self.fps)
        
        target_keywords = ["grifo", "tap", "handle", "abierto_1"] 
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

    class ConfusionMatrixModal(QDialog):
        def __init__(self, y_true, y_pred, class_name, parent=None):
            super().__init__(parent)
            self.setWindowTitle(f"Matriz de Confusión: {class_name}")
            self.setMinimumSize(500, 500)
            self.setStyleSheet("background-color: #1e1e1e; color: white;")
            
            layout = QVBoxLayout(self)
            
            # Calcular métricas usando sklearn
            cm = confusion_matrix(y_true, y_pred, labels=[1, 0]) # 1: Abierto, 0: Cerrado
            # Extraer valores (notar que labels=[1,0] cambia el orden estándar a TP, FN, FP, TN)
            tp, fn, fp, tn = cm.ravel()
            total = tp + tn + fp + fn
            
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

            # Fila 1: Realidad Abierto
            grid.addWidget(self._header_label("REALIDAD\n(CSV Abierto)"), 1, 0)
            grid.addWidget(self._cell_box(f"TP (Acierto)\n{tp}", f"{(tp/total*100):.1f}%", "#2ecc71"), 1, 1)
            grid.addWidget(self._cell_box(f"FN (Olvido)\n{fn}", f"{(fn/total*100):.1f}%", "#e74c3c"), 1, 2)

            # Fila 2: Realidad Cerrado
            grid.addWidget(self._header_label("REALIDAD\n(CSV Cerrado)"), 2, 0)
            grid.addWidget(self._cell_box(f"FP (Fantasma)\n{fp}", f"{(fp/total*100):.1f}%", "#e67e22"), 2, 1)
            grid.addWidget(self._cell_box(f"TN (Silencio)\n{tn}", f"{(tn/total*100):.1f}%", "#34495e"), 2, 2)

            layout.addLayout(grid)

            # Métricas resumen al pie
            metrics_text = (
                f"<b>Precisión:</b> {(tp/(tp+fp)*100 if (tp+fp)>0 else 0):.2f}% (Fiabilidad)<br>"
                f"<b>Recall:</b> {(tp/(tp+fn)*100 if (tp+fn)>0 else 0):.2f}% (Eficacia)<br>"
                f"<b>Total Frames Analizados:</b> {total}"
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


    def show_metrics(self):
        if self.total_frames > 0:
            stride = self.ai_config.get('stride', 1)
            modal = MetricsModal(self.bar_truth, self.detection_bars, self.total_frames, stride, self)
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