import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import math

class ImprovedPendulumTracker:
    def __init__(self, video_path, buffer_size=5):
        """
        Inicializar el tracker del péndulo mejorado
        
        Args:
            video_path (str): Ruta del video
            buffer_size (int): Tamaño del buffer para suavizar trayectoria
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.buffer_size = buffer_size
        
        # Variables para tracking
        self.positions = []
        self.angles = []
        self.frame_numbers = []
        self.timestamps = []
        
        # Para suavizado de trayectoria
        self.position_buffer = deque(maxlen=buffer_size)
        
        # Punto de referencia (pivot) del péndulo
        self.pivot_point = None
        
        # Variables para template matching
        self.template = None
        self.template_size = (40, 40)
        self.search_region_size = 100
        
        # Variables para motion detection
        self.prev_frame = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Obtener información del video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video cargado: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        
        self.metros_por_pixel = 2.0 / 400  # Ajusta 400 por la distancia en píxeles medida en tu video

    def set_pivot_point(self, x, y):
        """Establecer manualmente el punto de pivote del péndulo"""
        self.pivot_point = (x, y)
        print(f"Punto de pivote establecido en: ({x}, {y})")

    def create_template_from_region(self, frame, center, size=40):
        """
        Crear un template de la hamaca basado en una región alrededor del centro
        """
        x, y = center
        half_size = size // 2
        
        # Asegurar que el template esté dentro de los límites de la imagen
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(frame.shape[1], x + half_size)
        y2 = min(frame.shape[0], y + half_size)
        
        template = frame[y1:y2, x1:x2]
        
        if template.size > 0:
            self.template = template
            self.template_size = template.shape[:2]
            return True
        return False

    def detect_pendulum_motion_detection(self, frame):
        """
        Detectar la hamaca usando detección de movimiento mejorada
        """
        # Aplicar background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Limpiar la máscara con operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos en la máscara de movimiento
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filtrar contornos por área y proximidad al pivot
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100 and area < 5000:  # Filtrar por área razonable
                # Calcular centroide
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Verificar que esté a una distancia razonable del pivot
                    if self.pivot_point:
                        dist_to_pivot = np.sqrt((cx - self.pivot_point[0])**2 + 
                                              (cy - self.pivot_point[1])**2)
                        # La hamaca debe estar a una distancia razonable del pivot
                        if 50 < dist_to_pivot < 400:
                            valid_contours.append((contour, cx, cy, area, dist_to_pivot))
        
        if not valid_contours:
            return None
        
        # Seleccionar el contorno más probable (mayor área entre los válidos)
        valid_contours.sort(key=lambda x: x[3], reverse=True)
        _, cx, cy, _, _ = valid_contours[0]
        
        return (cx, cy)

    def detect_pendulum_template_matching(self, frame, last_position):
        """
        Detectar la hamaca usando template matching en una región de búsqueda
        """
        if self.template is None or last_position is None:
            return None
        
        # Definir región de búsqueda alrededor de la última posición conocida
        search_x = max(0, last_position[0] - self.search_region_size)
        search_y = max(0, last_position[1] - self.search_region_size)
        search_w = min(frame.shape[1] - search_x, 2 * self.search_region_size)
        search_h = min(frame.shape[0] - search_y, 2 * self.search_region_size)
        
        search_region = frame[search_y:search_y + search_h, search_x:search_x + search_w]
        
        if search_region.size == 0:
            return None
        
        # Convertir a escala de grises si es necesario
        if len(search_region.shape) == 3:
            search_region_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        else:
            search_region_gray = search_region
            
        if len(self.template.shape) == 3:
            template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = self.template
        
        # Template matching
        result = cv2.matchTemplate(search_region_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Umbral de confianza para template matching
        if max_val > 0.3:
            # Convertir coordenadas locales a globales
            match_x = search_x + max_loc[0] + template_gray.shape[1] // 2
            match_y = search_y + max_loc[1] + template_gray.shape[0] // 2
            return (match_x, match_y)
        
        return None

    def detect_pendulum_optical_flow(self, frame):
        """
        Detectar la hamaca usando optical flow (Lucas-Kanade)
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return None
        
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Parámetros para goodFeaturesToTrack
        feature_params = dict(maxCorners=100,
                            qualityLevel=0.01,
                            minDistance=10,
                            blockSize=7)
        
        # Parámetros para Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Detectar puntos característicos
        if self.pivot_point:
            # Buscar features en una región alrededor del punto esperado de la hamaca
            mask = np.zeros(self.prev_frame.shape, dtype=np.uint8)
            # Crear una máscara circular alrededor del área esperada de la hamaca
            cv2.circle(mask, self.pivot_point, 200, 255, -1)
            p0 = cv2.goodFeaturesToTrack(self.prev_frame, mask=mask, **feature_params)
        else:
            p0 = cv2.goodFeaturesToTrack(self.prev_frame, mask=None, **feature_params)
        
        if p0 is not None and len(p0) > 0:
            # Calcular optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, current_frame_gray, p0, None, **lk_params)
            
            # Seleccionar puntos válidos
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                if len(good_new) > 0:
                    # Filtrar puntos por movimiento significativo
                    movements = []
                    for new, old in zip(good_new, good_old):
                        movement = np.linalg.norm(new - old)
                        if movement > 2:  # Filtrar movimientos pequeños (ruido)
                            movements.append((new, movement))
                    
                    if movements:
                        # Tomar el punto con mayor movimiento (probablemente la hamaca)
                        movements.sort(key=lambda x: x[1], reverse=True)
                        best_point = movements[0][0]
                        
                        # Verificar que esté a distancia razonable del pivot
                        if self.pivot_point:
                            dist_to_pivot = np.linalg.norm(best_point - np.array(self.pivot_point))
                            if 50 < dist_to_pivot < 400:
                                self.prev_frame = current_frame_gray
                                return (int(best_point[0]), int(best_point[1]))
        
        self.prev_frame = current_frame_gray
        return None

    def detect_pendulum_hybrid(self, frame, frame_count):
        """
        Método híbrido que combina diferentes técnicas de detección
        """
        # Obtener la última posición válida
        last_valid_pos = None
        for pos in reversed(self.positions[-10:]):  # Revisar las últimas 10 posiciones
            if pos is not None:
                last_valid_pos = pos
                break
        
        detected_pos = None
        
        # Método 1: Template matching (si tenemos template y última posición)
        if self.template is not None and last_valid_pos is not None:
            detected_pos = self.detect_pendulum_template_matching(frame, last_valid_pos)
            if detected_pos:
                return detected_pos
        
        # Método 2: Motion detection
        motion_pos = self.detect_pendulum_motion_detection(frame)
        if motion_pos:
            # Validar con posición previa si existe
            if last_valid_pos is None:
                detected_pos = motion_pos
            else:
                # Verificar que la nueva posición no esté demasiado lejos de la anterior
                dist = np.sqrt((motion_pos[0] - last_valid_pos[0])**2 + 
                              (motion_pos[1] - last_valid_pos[1])**2)
                if dist < 100:  # Ajustar según la velocidad esperada de la hamaca
                    detected_pos = motion_pos
        
        # Método 3: Optical flow como respaldo
        if detected_pos is None:
            flow_pos = self.detect_pendulum_optical_flow(frame)
            if flow_pos:
                detected_pos = flow_pos
        
        return detected_pos

    def calculate_angle(self, position):
        """
        Calcular el ángulo del péndulo respecto a la vertical
        """
        if self.pivot_point is None or position is None:
            return None
        
        dx = position[0] - self.pivot_point[0]
        dy = position[1] - self.pivot_point[1]
        
        # Ángulo respecto a la vertical (0° = vertical hacia abajo)
        angle = math.atan2(dx, dy)
        return angle

    def smooth_position(self, position):
        """
        Suavizar la posición usando promedio móvil ponderado
        """
        if position is None:
            return None
        
        self.position_buffer.append(position)
        
        if len(self.position_buffer) == 1:
            return position
        
        # Promedio ponderado (más peso a posiciones recientes)
        weights = np.linspace(0.5, 1.0, len(self.position_buffer))
        weights = weights / weights.sum()
        
        avg_x = sum(pos[0] * w for pos, w in zip(self.position_buffer, weights))
        avg_y = sum(pos[1] * w for pos, w in zip(self.position_buffer, weights))
        
        return (int(avg_x), int(avg_y))

    def track_pendulum(self, show_video=True, use_motion_detection=True):
        """
        Realizar el tracking completo del péndulo con algoritmo mejorado
        """
        frame_count = 0
        
        # Configurar pivot point si no está establecido
        if self.pivot_point is None:
            ret, first_frame = self.cap.read()
            if ret:
                cv2.imshow('Seleccionar punto de pivote - Click y presiona ESC', first_frame)
                
                def mouse_callback(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        self.set_pivot_point(x, y)
                        cv2.circle(first_frame, (x, y), 5, (0, 0, 255), -1)
                        cv2.imshow('Seleccionar punto de pivote - Click y presiona ESC', first_frame)
                
                cv2.setMouseCallback('Seleccionar punto de pivote - Click y presiona ESC', mouse_callback)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Configurar template inicial
        ret, first_frame = self.cap.read()
        if ret and self.pivot_point:
            # Estimar posición inicial de la hamaca (debajo del pivot)
            initial_pos = (self.pivot_point[0], self.pivot_point[1] + 150)
            
            # Permitir al usuario seleccionar la posición inicial de la hamaca
            temp_frame = first_frame.copy()
            cv2.circle(temp_frame, initial_pos, 5, (0, 255, 0), -1)
            cv2.putText(temp_frame, 'Click en la hamaca para crear template - ESC para continuar', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Seleccionar hamaca', temp_frame)
            
            selected_pos = [initial_pos]
            
            def template_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    selected_pos[0] = (x, y)
                    temp_frame2 = first_frame.copy()
                    cv2.circle(temp_frame2, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(temp_frame2, 'Template seleccionado - Presiona ESC', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow('Seleccionar hamaca', temp_frame2)
            
            cv2.setMouseCallback('Seleccionar hamaca', template_callback)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Crear template inicial
            self.create_template_from_region(first_frame, selected_pos[0], 50)
        
        # Reiniciar video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detectar posición del péndulo usando método híbrido
            if use_motion_detection:
                position = self.detect_pendulum_hybrid(frame, frame_count)
            else:
                position = self.detect_pendulum_motion_detection(frame)
            
            # Suavizar posición
            smooth_pos = self.smooth_position(position)
            
            # Calcular ángulo
            angle = self.calculate_angle(smooth_pos)
            
            # Almacenar datos
            timestamp = frame_count / self.fps
            self.frame_numbers.append(frame_count)
            self.timestamps.append(timestamp)
            self.positions.append(smooth_pos)
            self.angles.append(angle)
            
            # Actualizar template ocasionalmente con detecciones válidas
            if smooth_pos and frame_count % 30 == 0:
                self.create_template_from_region(frame, smooth_pos, 50)
            
            # Visualización
            if show_video:
                display_frame = frame.copy()
                
                # Dibujar pivot
                if self.pivot_point:
                    cv2.circle(display_frame, self.pivot_point, 8, (255, 0, 0), -1)
                    cv2.putText(display_frame, 'PIVOT', 
                               (self.pivot_point[0]-20, self.pivot_point[1]-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Dibujar posición detectada
                if smooth_pos:
                    cv2.circle(display_frame, smooth_pos, 8, (0, 255, 0), -1)
                    
                    # Dibujar línea del péndulo
                    if self.pivot_point:
                        cv2.line(display_frame, self.pivot_point, smooth_pos, (0, 255, 255), 2)
                    
                    # Mostrar información
                    info_text = f"Frame: {frame_count}"
                    cv2.putText(display_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    if angle is not None:
                        angle_deg = math.degrees(angle)
                        angle_text = f"Angulo: {angle_deg:.1f}°"
                        cv2.putText(display_frame, angle_text, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Tracking del Pendulo', display_frame)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):  # Pausar
                    cv2.waitKey(0)
            
            frame_count += 1
            
            # Mostrar progreso
            if frame_count % 30 == 0:
                progress = (frame_count / self.total_frames) * 100
                valid_detections = sum(1 for pos in self.positions if pos is not None)
                detection_rate = (valid_detections / len(self.positions)) * 100 if self.positions else 0
                print(f"Progreso: {progress:.1f}% ({frame_count}/{self.total_frames}) - "
                      f"Detecciones válidas: {detection_rate:.1f}%")
        
        if show_video:
            cv2.destroyAllWindows()
        
        self.cap.release()
        print(f"Tracking completado. {len(self.positions)} frames procesados.")

    def get_tracking_data(self):
        """
        Obtener los datos del tracking como DataFrame
        """
        data = {
            'frame': self.frame_numbers,
            'timestamp': self.timestamps,
            'x_position': [pos[0] if pos else None for pos in self.positions],
            'y_position': [pos[1] if pos else None for pos in self.positions],
            'angle_rad': self.angles,
            'angle_deg': [math.degrees(ang) if ang else None for ang in self.angles]
        }
        
        return pd.DataFrame(data)

    def save_data(self, filename='pendulum_data.csv'):
        """
        Guardar los datos del tracking en un archivo CSV
        """
        df = self.get_tracking_data()
        df.to_csv(filename, index=False)
        print(f"Datos guardados en {filename}")

    def plot_results(self):
        """
        Generar gráficos de los resultados del tracking
        """
        df = self.get_tracking_data()
        valid_data = df.dropna()
        
        if len(valid_data) == 0:
            print("No hay datos válidos para graficar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis del Movimiento Pendular', fontsize=16)
        
        # Gráfico 1: Trayectoria
        axes[0, 0].scatter(valid_data['x_position'], valid_data['y_position'], 
                          c=valid_data['timestamp'], cmap='viridis', s=2)
        axes[0, 0].set_xlabel('Posición X (píxeles)')
        axes[0, 0].set_ylabel('Posición Y (píxeles)')
        axes[0, 0].set_title('Trayectoria del Péndulo')
        axes[0, 0].invert_yaxis()
        
        # Gráfico 2: Ángulo vs Tiempo
        axes[0, 1].plot(valid_data['timestamp'], valid_data['angle_deg'], linewidth=1)
        axes[0, 1].set_xlabel('Tiempo (s)')
        axes[0, 1].set_ylabel('Ángulo (grados)')
        axes[0, 1].set_title('Ángulo vs Tiempo')
        axes[0, 1].grid(True)
        
        # Gráfico 3: Posición X vs Tiempo
        axes[1, 0].plot(valid_data['timestamp'], valid_data['x_position'], linewidth=1)
        axes[1, 0].set_xlabel('Tiempo (s)')
        axes[1, 0].set_ylabel('Posición X (píxeles)')
        axes[1, 0].set_title('Posición Horizontal vs Tiempo')
        axes[1, 0].grid(True)
        
        # Gráfico 4: Distribución de ángulos
        axes[1, 1].hist(valid_data['angle_deg'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Ángulo (grados)')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución de Ángulos')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

    def calculate_period(self):
        """
        Calcular el período del péndulo usando análisis de frecuencias
        """
        df = self.get_tracking_data()
        valid_data = df.dropna()
        
        if len(valid_data) < 100:
            print("No hay suficientes datos para calcular el período")
            return None
        
        angles = valid_data['angle_deg'].values
        times = valid_data['timestamp'].values
        
        # Interpolar para tener muestras uniformes
        dt = np.mean(np.diff(times))
        t_uniform = np.arange(times[0], times[-1], dt)
        angles_uniform = np.interp(t_uniform, times, angles)
        
        # Calcular FFT
        fft = np.fft.fft(angles_uniform - np.mean(angles_uniform))
        freqs = np.fft.fftfreq(len(angles_uniform), dt)
        
        # Encontrar la frecuencia dominante (excluyendo DC)
        magnitude = np.abs(fft)
        dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        dominant_freq = np.abs(freqs[dominant_freq_idx])
        
        if dominant_freq > 0:
            period = 1.0 / dominant_freq
            print(f"Período estimado: {period:.3f} segundos")
            print(f"Frecuencia dominante: {dominant_freq:.3f} Hz")
            return period
        
        return None


def main():
    # Crear el tracker mejorado
    tracker = ImprovedPendulumTracker('./Videos/hamaca-juli-recortado.mp4')
    
    print("Iniciando tracking mejorado para hamaca...")
    print("Instrucciones:")
    print("1. Selecciona el punto de pivote (donde se sujeta la hamaca)")
    print("2. Selecciona un punto en la hamaca para crear el template")
    print("3. El algoritmo combinará detección de movimiento, template matching y optical flow")
    
    # Ejecutar tracking con algoritmo híbrido mejorado
    tracker.track_pendulum(show_video=True, use_motion_detection=True)
    
    # Obtener y mostrar estadísticas
    data = tracker.get_tracking_data()
    valid_data = data.dropna()
    
    print(f"\n=== ESTADÍSTICAS FINALES ===")
    print(f"Total de frames: {len(data)}")
    print(f"Frames con detección válida: {len(valid_data)}")
    print(f"Tasa de éxito: {(len(valid_data)/len(data)*100):.1f}%")
    
    if len(valid_data) > 0:
        print(f"Ángulo promedio: {valid_data['angle_deg'].mean():.2f}°")
        print(f"Ángulo máximo: {valid_data['angle_deg'].max():.2f}°")
        print(f"Ángulo mínimo: {valid_data['angle_deg'].min():.2f}°")
        print(f"Amplitud del movimiento: {valid_data['angle_deg'].max() - valid_data['angle_deg'].min():.2f}°")
        
        # Guardar datos y generar gráficos
        tracker.save_data('datos_hamaca_mejorado.csv')
        tracker.plot_results()
        
        # Calcular período
        period = tracker.calculate_period()
        if period:
            print(f"Período estimado: {period:.3f} segundos")
    else:
        print("No se pudieron obtener datos válidos.")
        print("Sugerencias:")
        print("1. Asegúrate de que el punto de pivote esté bien seleccionado")
        print("2. Verifica que la hamaca tenga buen contraste con el fondo")
        print("3. La iluminación debe ser uniforme")
        print("4. Intenta seleccionar un punto más distintivo de la hamaca para el template")


if __name__ == "__main__":
    main()