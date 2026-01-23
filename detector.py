import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from ultralytics import YOLO

class AnalysisThread(QThread):
    frame_ready = Signal(QImage)
    detection_event = Signal(int, dict)

    def __init__(self, video_path, model_path, config, start_frame=0):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.config = config
        self.start_frame = start_frame
        
        # Banderas de control
        self.running = True
        self.paused = False       # Nueva bandera para pausar
        self.target_frame = -1    # Indica si hay que saltar a un frame concreto
        
        self.fps = 15.0

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 15.0

        # Posicionamiento inicial
        frame_idx = self.start_frame
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
        stride = self.config.get('stride', 1)
        iou_threshold = self.config.get('iou', 0.45)
        class_map = self.config.get('class_map', {})
        class_names = self.model.names 

        while self.running and cap.isOpened():
            # 1. GESTIÓN DE SALTO (SEEK)
            # Si desde la interfaz nos dicen "vete al frame 500", lo hacemos aquí
            if self.target_frame >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.target_frame)
                frame_idx = self.target_frame
                self.target_frame = -1 # Reseteamos la orden de salto
            
            # 2. GESTIÓN DE PAUSA
            # Si está pausado, el hilo duerme un poco para no quemar CPU y vuelve a comprobar
            if self.paused:
                self.msleep(100) # Dormir 100ms
                continue

            # 3. PROCESAMIENTO NORMAL
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % stride == 0:
                # Inferencia con confianza muy baja para filtrar después
                results = self.model(frame, conf=0.05, iou=iou_threshold, verbose=False)[0]
                
                frame_results = {name: False for name in class_names.values()}
                
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label = class_names[cls_id]
                    confidence = float(box.conf[0])
                    required_conf = class_map.get(label, {}).get('conf', 0.40)
                    
                    if confidence >= required_conf:
                        frame_results[label] = True

                self.detection_event.emit(frame_idx, frame_results)
                
                rgb = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, w*ch, QImage.Format_RGB888)
                self.frame_ready.emit(img)
            
            frame_idx += 1
            
        cap.release()

    def stop(self):
        self.running = False
        self.wait() # Esperar a que cierre bien

    def set_paused(self, paused):
        """Método para pausar o reanudar desde fuera"""
        self.paused = paused

    def seek(self, frame_num):
        """Método para ordenar un salto de frame"""
        self.target_frame = frame_num