import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

print(cv2.__version__)

# Abrir el video
cap = cv2.VideoCapture(r"./Videos/clemen-tieso.mp4")

# Obtener FPS del video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS del video: {fps}")

# ================== PARÁMETROS FÍSICOS ==================
# Masa (kg) de la "carga" 
# Si esta la hamaca sola 2kg
# Si esta clemente 72kg
m = 72.0
# Aceleración gravitatoria (m/s^2)
g = 9.81

# Factor de conversión temporal
time_per_frame = 1.0 / fps  # segundos por frame
print(f"Factor temporal: 1 frame = {time_per_frame:.4f} segundos")

# Leer el primer frame
ret, frame = cap.read()
if not ret:
    print("No se pudo leer el video")
    exit()

# Seleccionar el objeto (hamaca) con el mouse
bbox = cv2.selectROI("Selecciona la hamaca", frame, False)
cv2.destroyWindow("Selecciona la hamaca")  # Cerrar ventana de selección

# Seleccionar el punto de suspensión de la hamaca
print("\nAhora selecciona el punto de suspensión de la hamaca (donde está colgada)")
origin_roi = cv2.selectROI("Selecciona punto de suspensión", frame, False)
cv2.destroyWindow("Selecciona punto de suspensión")

# Calcular el origen real (punto de suspensión)
x_origin = origin_roi[0] + origin_roi[2] / 2
y_origin = origin_roi[1] + origin_roi[3] / 2

print(f"Punto de suspensión seleccionado: ({x_origin:.1f}, {y_origin:.1f})")

# Crear el tracker CSRT
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

positions = []  # Lista para guardar posiciones del centro

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Actualizar tracker
    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        # Dibujar rectángulo
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Guardar posición del centro
        center_x = x + w / 2
        center_y = y + h / 2
        positions.append((center_x, center_y))
    else:
        cv2.putText(frame, "Tracking fallido", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar frame actualizado
    cv2.imshow("Tracking", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Función de suavizado con ventana móvil
def smooth_data(data, window_size=5):
    """Aplica suavizado de media móvil con ventana de tamaño window_size"""
    if len(data) < window_size:
        return data
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    # Rellenar los bordes para mantener el tamaño original
    pad_start = (window_size - 1) // 2
    pad_end = window_size - 1 - pad_start
    return np.concatenate([data[:pad_start], smoothed, data[-pad_end:] if pad_end > 0 else []])

# Procesar las posiciones para graficar
if len(positions) > 1:
    pos_array = np.array(positions)
    x_pos_raw = pos_array[:, 0]
    y_pos_raw = pos_array[:, 1]
    
    # Aplicar suavizado a las posiciones
    x_pos = smooth_data(x_pos_raw, window_size=5)
    y_pos = smooth_data(y_pos_raw, window_size=5)

    # Función para encontrar el centro de rotación automáticamente
    def find_rotation_center(x_pos, y_pos):
        best_center = None
        min_radius_std = float('inf')
        
        # Probar diferentes puntos en una grilla alrededor del punto manual
        x_search = np.linspace(x_origin - 100, x_origin + 100, 15)
        y_search = np.linspace(y_origin - 100, y_origin + 100, 15)
        
        for test_x in x_search:
            for test_y in y_search:
                test_r = np.sqrt((x_pos - test_x)**2 + (y_pos - test_y)**2)
                radius_std = np.std(test_r)
                
                if radius_std < min_radius_std:
                    min_radius_std = radius_std
                    best_center = (test_x, test_y)
        
        return best_center, min_radius_std

    # Comparar el origen manual vs automático
    print(f"\n=== ANÁLISIS DEL CENTRO DE ROTACIÓN ===")
    print(f"Origen manual seleccionado: ({x_origin:.1f}, {y_origin:.1f})")
    
    # Calcular estadísticas con el origen manual
    x_rel_manual = x_pos - x_origin
    y_rel_manual = y_pos - y_origin
    r_manual = np.sqrt(x_rel_manual**2 + y_rel_manual**2)
    manual_std = np.std(r_manual)
    
    # Buscar el mejor centro automáticamente
    auto_center, auto_std = find_rotation_center(x_pos, y_pos)
    x_origin_auto, y_origin_auto = auto_center
    
    print(f"Origen automático encontrado: ({x_origin_auto:.1f}, {y_origin_auto:.1f})")
    print(f"Desviación estándar del radio (manual): {manual_std:.2f} pixeles")
    print(f"Desviación estándar del radio (automático): {auto_std:.2f} pixeles")
    
    # Usar el mejor origen (el que tiene menor desviación estándar)
    if auto_std < manual_std:
        print("✓ Usando origen automático (menor variación de radio)")
        x_origin_final = x_origin_auto
        y_origin_final = y_origin_auto
    else:
        print("✓ Usando origen manual (ya es óptimo)")
        x_origin_final = x_origin
        y_origin_final = y_origin
    
    # Coordenadas relativas al origen final
    x_rel = x_pos - x_origin_final
    y_rel = y_pos - y_origin_final

    # Calcular radio y ángulo ANTES de usar en los gráficos
    r = np.sqrt(x_rel**2 + y_rel**2)
    theta = np.arctan2(y_rel, x_rel)

    # ================== CALIBRACIÓN DE ESCALA ==================
    # El radio de la hamaca cuando está vertical (velocidad máxima) es 2.32 m
    radio_real = 2.32  # metros
    radio_promedio_pixels = np.mean(r)
    
    # Calcular factor de escala: píxeles → metros
    meters_per_pixel = radio_real / radio_promedio_pixels
    pixels_per_meter = 1.0 / meters_per_pixel
    
    print(f"\n=== CALIBRACIÓN DE ESCALA ===")
    print(f"Radio promedio en píxeles: {radio_promedio_pixels:.1f} px")
    print(f"Radio real de la hamaca: {radio_real} m")
    print(f"Factor de conversión: 1 píxel = {meters_per_pixel:.6f} metros")
    print(f"Factor de conversión: 1 metro = {pixels_per_meter:.1f} píxeles")

    # ================== CONVERTIR A UNIDADES REALES ==================
    # Posiciones en metros
    x_pos_m = x_pos * meters_per_pixel
    y_pos_m = y_pos * meters_per_pixel
    r_m = r * meters_per_pixel
    
    # Tiempos en segundos
    time_pos = np.arange(len(x_pos)) * time_per_frame
    time_vel = np.arange(1, len(x_pos)) * time_per_frame
    time_acc = np.arange(2, len(x_pos)) * time_per_frame

    # Calcular velocidad (diferencias entre posiciones)
    vx_raw = np.diff(x_pos)
    vy_raw = np.diff(y_pos)
    
    # Suavizar velocidades
    vx = smooth_data(vx_raw, window_size=5)
    vy = smooth_data(vy_raw, window_size=5)
    
    # Convertir velocidades a m/s
    vx_ms = vx * meters_per_pixel / time_per_frame
    vy_ms = vy * meters_per_pixel / time_per_frame
    
    # Calcular aceleración (segunda derivada)
    ax_raw = np.diff(vx)
    ay_raw = np.diff(vy)
    
    # Suavizar aceleraciones
    ax = smooth_data(ax_raw, window_size=5)
    ay = smooth_data(ay_raw, window_size=5)
    
    # Convertir aceleraciones a m/s²
    ax_ms2 = ax * meters_per_pixel / (time_per_frame ** 2)
    ay_ms2 = ay * meters_per_pixel / (time_per_frame ** 2)

    frames_pos = np.arange(len(x_pos))         # Para posiciones
    frames_vel = np.arange(1, len(x_pos))      # Para velocidades
    frames_acc = np.arange(2, len(x_pos))      # Para aceleraciones

    # Calcular x_origin_m y y_origin_m para el resumen
    x_origin_m = x_origin_final * meters_per_pixel
    y_origin_m = y_origin_final * meters_per_pixel

    # Ajustar tamaños de fuente para evitar solapamientos en pantallas pequeñas
    plt.rcParams.update({
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })

    # ================== COORDENADAS CARTESIANAS ==================
    plt.figure(figsize=(12, 14), constrained_layout=True)
    
    plt.subplot(3, 2, 1)
    plt.plot(time_pos, x_pos_m, label='Posición X', color='b')
    plt.plot(time_pos, y_pos_m, label='Posición Y', color='g')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición (m)')
    plt.title('Posición Cartesiana vs Tiempo')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(time_vel, vx_ms, label='Velocidad en X', color='r')
    plt.plot(time_vel, vy_ms, label='Velocidad en Y', color='m')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad (m/s)')
    plt.title('Velocidad Cartesiana vs Tiempo')
    plt.legend()
    plt.grid(True)

    # ================== COORDENADAS POLARES ==================
    # Debug: Análisis de los ángulos
    print(f"\n=== DEBUG COORDENADAS POLARES ===")
    print(f"Radio promedio: {np.mean(r_m):.3f} ± {np.std(r_m):.3f} metros")
    print(f"Rango de ángulos: {np.degrees(np.min(theta)):.1f}° a {np.degrees(np.max(theta)):.1f}°")
    print(f"Variación angular total: {np.degrees(np.max(theta) - np.min(theta)):.1f}°")
    
    # Manejar discontinuidades en theta (salto de -π a π)
    theta_unwrapped = np.unwrap(theta)
    
    # Velocidades polares
    vr_raw = np.diff(r)
    vtheta_raw = np.diff(theta_unwrapped)
    
    # Suavizar velocidades polares
    vr = smooth_data(vr_raw, window_size=5)
    vtheta = smooth_data(vtheta_raw, window_size=5)
    
    # Convertir a unidades reales
    vr_ms = vr * meters_per_pixel / time_per_frame
    vtheta_rads = vtheta / time_per_frame  # rad/s

    # ================== FUERZAS (DINÁMICA) ==================
    # Longitud efectiva del péndulo (radio medio)
    L = np.mean(r_m)
    # Alinear ángulo con velocidades (descartar primer punto para coincidir tamaños)
    theta_mid = theta_unwrapped[1:]  # corresponde a frames_vel
    omega = vtheta_rads  # ya en rad/s, tamaño len(frames_vel)
    # Tensión: T = m*g*cos(theta) + m*L*omega^2
    tension = m * g * np.cos(theta_mid) + m * L * omega**2
    peso = m * g  # constante
    
    print(f"Velocidad angular máxima: {np.max(np.abs(vtheta_rads)):.3f} rad/s")
    print(f"Velocidad angular promedio: {np.mean(np.abs(vtheta_rads)):.3f} rad/s")
    
    plt.subplot(3, 2, 3)
    plt.plot(time_pos, r_m, 'purple', label='Radio r')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Radio (m)')
    plt.title('Radio vs Tiempo (Coordenadas Polares)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.plot(time_pos, np.degrees(theta_unwrapped), 'orange', label='Ángulo θ')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Ángulo (grados)')
    plt.title('Ángulo vs Tiempo (Coordenadas Polares)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 5)
    plt.plot(time_vel, vtheta_rads, 'purple', label='Velocidad angular')
    plt.plot(time_vel, vr_ms, 'orange', label='Velocidad radial')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad angular (rad/s) / radial (m/s)')
    plt.title('Velocidades Polares vs Tiempo')
    plt.legend()
    plt.grid(True)

    # ================== COORDENADAS INTRÍNSECAS ==================
    # Calcular velocidad escalar y aceleración
    v_magnitude_raw = np.sqrt(vx_ms**2 + vy_ms**2)
    v_magnitude = smooth_data(v_magnitude_raw, window_size=5)
    
    # Vector tangente unitario (dirección de la velocidad)
    v_mag_safe = np.where(v_magnitude > 1e-6, v_magnitude, 1e-6)  # Evitar división por cero
    t_x = vx_ms / v_mag_safe
    t_y = vy_ms / v_mag_safe
    
    # Vector normal unitario (perpendicular al tangente, hacia la izquierda)
    n_x = -t_y
    n_y = t_x
    
    # Componentes intrínsecas de la aceleración
    if len(ax_ms2) > 0 and len(ay_ms2) > 0:
        # Para las aceleraciones necesitamos los vectores tangente en los puntos correspondientes
        # ax, ay tienen tamaño len(vx)-1 = len(x_pos)-2
        # t_x, t_y tienen tamaño len(x_pos)-1
        # Necesitamos alinear los índices correctamente
        
        # Usar los vectores tangente en los puntos donde tenemos aceleración
        t_x_acc = t_x[1:]  # Tomar desde el segundo elemento
        t_y_acc = t_y[1:]  # Tomar desde el segundo elemento
        n_x_acc = n_x[1:]  # Tomar desde el segundo elemento
        n_y_acc = n_y[1:]  # Tomar desde el segundo elemento
        
        # Aceleración tangencial (componente en dirección del movimiento)
        a_tangential_raw = ax_ms2 * t_x_acc + ay_ms2 * t_y_acc
        a_tangential = smooth_data(a_tangential_raw, window_size=5)
        
        # Aceleración normal (componente perpendicular al movimiento)
        a_normal_raw = ax_ms2 * n_x_acc + ay_ms2 * n_y_acc
        a_normal = smooth_data(a_normal_raw, window_size=5)
        
        # Radio de curvatura
        v_mag_curvature = v_magnitude[1:]  # Velocidad en los puntos donde tenemos aceleración
        radius_curvature = np.where(np.abs(a_normal) > 1e-6, 
                                  v_mag_curvature**2 / np.abs(a_normal), 
                                  np.inf)
    else:
        a_tangential = np.array([])
        a_normal = np.array([])
        radius_curvature = np.array([])
    
    if len(a_tangential) > 0:
        plt.subplot(3, 2, 6)
        plt.plot(time_acc, a_tangential, 'red', label='Aceleración tangencial')
        plt.plot(time_acc, a_normal, 'blue', label='Aceleración normal')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Aceleración (m/s²)')
        plt.title('Aceleraciones Intrínsecas vs Tiempo')
        plt.legend()
        plt.grid(True)

    # Layout automático con separación adecuada entre filas/columnas
    plt.show()

    # ================== DIAGRAMA DE CUERPO LIBRE ==================
    # Elegir frame de máxima velocidad angular para FBD
    idx_max_v = np.argmax(np.abs(omega))
    theta_fbd = theta_mid[idx_max_v]
    omega_fbd = omega[idx_max_v]
    tension_fbd = tension[idx_max_v]

    # Vectores en sistema local (bob en origen)
    # Dirección hacia el pivote (tensión): vector radial negativo
    dir_tension = -np.array([np.cos(theta_fbd), np.sin(theta_fbd)])  # pivote hacia bob era (cos, sin)
    dir_peso = np.array([0.0, 1.0])  # y positivo hacia abajo en nuestra convención

    # Escalado para visualización
    T_max = np.max(tension)
    scale = 1.0 / T_max
    vec_T = dir_tension * tension_fbd * scale * L
    vec_W = dir_peso * peso * scale * L

    plt.figure(figsize=(5,5))
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.quiver(0, 0, vec_T[0], vec_T[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Tensión')
    plt.quiver(0, 0, vec_W[0], vec_W[1], angles='xy', scale_units='xy', scale=1, color='green', label='Peso')
    # Dibujar varilla
    bob_pos = np.array([np.sin(theta_fbd)*L*0.4, np.cos(theta_fbd)*L*0.4])
    plt.plot([0, bob_pos[0]], [0, bob_pos[1]], 'k-', linewidth=2)
    plt.scatter([0], [0], color='red', s=60, label='Masa')
    plt.title('Diagrama de Cuerpo Libre (frame velocidad máx)')
    plt.xlabel('Eje X (relativo)')
    plt.ylabel('Eje Y (relativo)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # ================== SIMULACIÓN FUERZAS SOBRE TRAYECTORIA ==================
    # Submuestreo para flechas (no saturar)
    step = max(1, len(theta_mid)//40)
    sample_indices = np.arange(0, len(theta_mid), step)
    x_rel_m = (x_pos_m - x_origin_m)
    y_rel_m = (y_pos_m - y_origin_m)
    x_rel_mid = x_rel_m[1:]  # alineado con theta_mid
    y_rel_mid = y_rel_m[1:]

    plt.figure(figsize=(12,6), constrained_layout=True)
    ax1 = plt.subplot(1,2,1)
    # Trayectoria (relativa al pivote)
    ax1.plot(x_rel_m, y_rel_m, color='gray', linewidth=1, label='Trayectoria real')
    # Pivot
    ax1.scatter([0],[0], color='black', s=30, label='Pivote')
    # Flechas de fuerzas en puntos muestreados
    for i in sample_indices:
        th = theta_mid[i]
        # Posición del bob en este índice
        bx = x_rel_mid[i]
        by = y_rel_mid[i]
        # Direcciones
        dir_T = -np.array([np.cos(th), np.sin(th)])
        dir_W = np.array([0.0, 1.0])
        # Magnitudes escaladas
        T_mag = tension[i] * scale * L * 0.6
        W_mag = peso * scale * L * 0.6
        ax1.quiver(bx, by, dir_T[0]*T_mag, dir_T[1]*T_mag, angles='xy', scale_units='xy', scale=1, color='blue')
        ax1.quiver(bx, by, dir_W[0]*W_mag, dir_W[1]*W_mag, angles='xy', scale_units='xy', scale=1, color='green')
    ax1.set_title('Trayectoria y Fuerzas (submuestreadas)')
    ax1.set_xlabel('X relativa (m)')
    ax1.set_ylabel('Y relativa (m)')
    ax1.legend(loc='upper right')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(1,2,2)
    ax2.plot(time_vel, tension, color='blue', label='Tensión (N)')
    ax2.axhline(peso, color='green', linestyle='--', label='Peso (N)')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Fuerza (N)')
    ax2.set_title('Evolución de Fuerzas')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.show()
    
    # ================== RESUMEN DE DATOS ==================
    print("\n=== RESUMEN DEL ANÁLISIS ===")
    print(f"Número total de frames analizados: {len(positions)}")
    print(f"Duración total del análisis: {len(positions) * time_per_frame:.2f} segundos")
    print(f"Origen final utilizado: ({x_origin_final:.1f}, {y_origin_final:.1f}) px = ({x_origin_m:.3f}, {y_origin_m:.3f}) m")
    print(f"Radio promedio: {np.mean(r_m):.3f} ± {np.std(r_m):.3f} metros")
    print(f"Radio mínimo: {np.min(r_m):.3f} metros")
    print(f"Radio máximo: {np.max(r_m):.3f} metros")
    print(f"Variación del radio: {((np.max(r_m) - np.min(r_m)) / np.mean(r_m) * 100):.1f}%")
    print(f"Velocidad escalar promedio: {np.mean(v_magnitude):.3f} m/s")
    print(f"Velocidad angular máxima: {np.max(np.abs(vtheta_rads)):.3f} rad/s = {np.degrees(np.max(np.abs(vtheta_rads))):.3f}°/s")
    
    # Análisis del período de oscilación
    peaks, _ = find_peaks(np.degrees(theta_unwrapped), height=None, distance=20)
    if len(peaks) > 1:
        periodo_frames = np.mean(np.diff(peaks)) * 2  # *2 porque son semi-oscilaciones
        periodo_segundos = periodo_frames * time_per_frame
        frecuencia = 1.0 / periodo_segundos
        print(f"Período estimado: {periodo_segundos:.2f} segundos")
        print(f"Frecuencia: {frecuencia:.3f} Hz")
    
    if len(radius_curvature) > 0:
        radius_finite_values = radius_curvature[np.isfinite(radius_curvature)]
        if len(radius_finite_values) > 0:
            print(f"Radio de curvatura promedio: {np.mean(radius_finite_values):.3f} metros")

else:
    print("No se detectaron suficientes posiciones para graficar.")
