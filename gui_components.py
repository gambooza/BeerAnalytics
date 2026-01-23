from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QBrush
from PySide6.QtCore import Qt

class DetectionBar(QWidget):
    def __init__(self, color_hex, label, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(30)
        self.base_color = QColor(color_hex)
        self.label_text = label
        
        # EL DATO CLAVE: Aquí guardamos la historia (0 o 1) de cada frame
        self.buffer = [] 
        self.total_frames = 0
        self.current_frame = 0

    def init_buffer(self, total_frames):
        """Inicializa la lista vacía con ceros"""
        self.total_frames = total_frames
        # Creamos una lista llena de ceros (False)
        self.buffer = [0] * total_frames
        self.update() # Forzar repintado

    def mark_detection(self, frame_idx, is_detected):
        """Guarda el resultado en la lista y pide repintar"""
        if 0 <= frame_idx < len(self.buffer):
            # Guardamos 1 si detectado, 0 si no
            self.buffer[frame_idx] = 1 if is_detected else 0
            self.update()

    def set_current_frame(self, frame_idx):
        self.current_frame = frame_idx
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        
        # 1. Fondo (Gris oscuro)
        painter.fillRect(self.rect(), QColor("#222"))
        
        # 2. Dibujar Etiqueta (C, G, TRUTH...)
        painter.setPen(Qt.white)
        painter.drawText(5, 20, self.label_text)
        
        # Si no hay datos, terminamos
        if not self.buffer:
            return

        # 3. Dibujar las detecciones
        # Calculamos cuánto mide cada frame en píxeles
        w = self.width()
        h = self.height()
        # Dejamos 40px a la izquierda para el texto
        draw_area_x = 40 
        draw_area_w = w - draw_area_x
        
        if self.total_frames > 0:
            step = draw_area_w / self.total_frames
            
            # Configuramos el color de la barra
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(self.base_color))
            
            # Recorremos el buffer y pintamos donde haya un 1
            # OPTIMIZACIÓN: Si hay muchos frames, pintar rectángulos agrupados
            # Para simplificar, pintamos bloques
            for i, val in enumerate(self.buffer):
                if val == 1:
                    x_pos = draw_area_x + (i * step)
                    # Pintamos un bloque de 1 frame de ancho (o mínimo 1px)
                    block_w = max(1, step) 
                    # ceil para evitar huecos en videos largos
                    import math
                    block_w = math.ceil(step)
                    
                    painter.drawRect(int(x_pos), 0, int(block_w), h)

            # 4. Dibujar cursor de reproducción (Línea blanca vertical)
            cursor_x = draw_area_x + (self.current_frame * step)
            painter.setPen(QColor(255, 255, 255, 150)) # Blanco semitransparente
            painter.drawLine(int(cursor_x), 0, int(cursor_x), h)