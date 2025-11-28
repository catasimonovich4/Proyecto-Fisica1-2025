import cv2
import numpy as np
import matplotlib.pyplot as plt

def seleccionar_color_manual(frame, nombre_color):
    """Permite al usuario seleccionar un punto y extrae su color HSV"""
    print(f"\n=== Seleccionar {nombre_color} ===")
    print("Haz clic en el parche de color. Presiona ESC para cancelar.")
    
    color_seleccionado = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal color_seleccionado
        if event == cv2.EVENT_LBUTTONDOWN:
            # Extraer color HSV del punto clickeado
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[y, x]
            # Convertir a int para evitar overflow
            color_seleccionado = (int(h), int(s), int(v))
            
            # Mostrar información
            print(f"Punto seleccionado: ({x}, {y})")
            print(f"Color HSV: H={h}, S={s}, V={v}")
            
            # Dibujar círculo en el punto seleccionado
            frame_temp = frame.copy()
            cv2.circle(frame_temp, (x, y), 10, (0, 255, 0), 2)
            cv2.putText(frame_temp, f"H:{h} S:{s} V:{v}", (x+15, y-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Seleccionar Color', frame_temp)
    
    cv2.namedWindow('Seleccionar Color')
    cv2.setMouseCallback('Seleccionar Color', mouse_callback)
    cv2.imshow('Seleccionar Color', frame)
    
    while color_seleccionado is None:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para cancelar
            cv2.destroyWindow('Seleccionar Color')
            return None
    
    cv2.waitKey(500)  # Pequeña pausa para ver la selección
    cv2.destroyWindow('Seleccionar Color')
    
    return color_seleccionado

def crear_rango_desde_color(h, s, v, tolerancia_h=15, tolerancia_s=50, tolerancia_v=50):
    """Crea un rango HSV con tolerancia a partir de un color base"""
    # Para el canal H (que va de 0-179 en OpenCV)
    h_min = max(0, h - tolerancia_h)
    h_max = min(179, h + tolerancia_h)
    
    # Para S y V
    s_min = max(0, s - tolerancia_s)
    s_max = min(255, s + tolerancia_s)
    v_min = max(0, v - tolerancia_v)
    v_max = min(255, v + tolerancia_v)
    
    return {
        'lower': np.array([h_min, s_min, v_min]),
        'upper': np.array([h_max, s_max, v_max]),
        'color_base': (h, s, v)
    }

def detectar_parche(frame, rango_color):
    """Detecta un parche de color usando el rango especificado"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, rango_color['lower'], rango_color['upper'])
    
    # Limpiar ruido
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Tomar el contorno más grande
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 50:  # Filtrar contornos muy pequeños
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy), mask
    
    return None, mask

# Abrir video
cap = cv2.VideoCapture(r"./Videos/clemen-normal.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS del video: {fps}")

# Leer el primer frame para selección de colores
ret, first_frame = cap.read()
if not ret:
    print("No se pudo leer el video")
    exit()

# Seleccionar el punto de suspensión (pivote) de la hamaca
print("\n" + "="*50)
print("SELECCIÓN DEL PUNTO DE SUSPENSIÓN")
print("="*50)
print("Selecciona el punto de suspensión de la hamaca (donde está colgada)")
origin_roi = cv2.selectROI("Selecciona punto de suspension", first_frame, False)
cv2.destroyWindow("Selecciona punto de suspension")

# Calcular el origen real (punto de suspensión)
x_pivot = origin_roi[0] + origin_roi[2] / 2
y_pivot = origin_roi[1] + origin_roi[3] / 2
print(f"Punto de suspensión seleccionado: ({x_pivot:.1f}, {y_pivot:.1f})")

# Permitir al usuario seleccionar los colores
print("\n" + "="*50)
print("SELECCIÓN DE COLORES")
print("="*50)
print("Seleccionarás dos parches de color en el primer frame del video.")
print("Haz clic en el centro de cada parche cuando se te indique.")

# Seleccionar primer color (celeste)
color_celeste_hsv = seleccionar_color_manual(first_frame, "PARCHE CELESTE")
if color_celeste_hsv is None:
    print("Selección cancelada")
    exit()

# Seleccionar segundo color (rojo oscuro)
color_rojo_hsv = seleccionar_color_manual(first_frame, "PARCHE ROJO OSCURO (HAMACA)")
if color_rojo_hsv is None:
    print("Selección cancelada")
    exit()

# Crear rangos de detección
print("\n" + "="*50)
print("CONFIGURACIÓN DE RANGOS")
print("="*50)

RANGOS_COLOR = {
    'celeste': crear_rango_desde_color(*color_celeste_hsv),
    'rojo_oscuro': crear_rango_desde_color(*color_rojo_hsv)
}

print(f"\nCeleste - Base: {color_celeste_hsv}")
print(f"  Rango: {RANGOS_COLOR['celeste']['lower']} a {RANGOS_COLOR['celeste']['upper']}")
print(f"\nRojo Oscuro - Base: {color_rojo_hsv}")
print(f"  Rango: {RANGOS_COLOR['rojo_oscuro']['lower']} a {RANGOS_COLOR['rojo_oscuro']['upper']}")

# Resetear video al inicio
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Almacenar posiciones
posiciones = {
    'celeste': [],      # Brazo/tobillo
    'rojo_oscuro': []   # Hamaca
}

frame_count = 0

print("\n" + "="*50)
print("PROCESAMIENTO DE VIDEO")
print("="*50)
print("Presiona 'q' para salir, 's' para pausar y ver máscaras")
print("Detectando parches...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_display = frame.copy()
    
    # Detectar cada parche usando los rangos seleccionados
    detecciones_frame = {}
    for color in ['celeste', 'rojo_oscuro']:
        centroid, mask = detectar_parche(frame, RANGOS_COLOR[color])
        detecciones_frame[color] = (centroid, mask)
        
        if centroid:
            posiciones[color].append(centroid)
            
            # Dibujar en el frame
            color_bgr = (255, 255, 0) if color == 'celeste' else (0, 0, 139)  # Cyan para celeste, rojo oscuro
            cv2.circle(frame_display, centroid, 8, color_bgr, -1)
            cv2.putText(frame_display, color.upper().replace('_', ' '), 
                       (centroid[0] + 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            # Si no se detecta, agregar None para mantener índices alineados
            posiciones[color].append(None)
    
    # Mostrar
    cv2.imshow('Detección de Parches', frame_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Pausar y mostrar máscaras individuales
        print(f"\nFrame {frame_count} pausado - Mostrando máscaras")
        for color in ['celeste', 'rojo_oscuro']:
            _, mask = detecciones_frame[color]
            cv2.imshow(f'Máscara {color}', mask)
        cv2.waitKey(0)
        for color in ['celeste', 'rojo_oscuro']:
            cv2.destroyWindow(f'Máscara {color}')
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Procesados {frame_count} frames...")

cap.release()
cv2.destroyAllWindows()

print(f"\nProcesamiento completado. Total frames: {frame_count}")

# ================== CALIBRACIÓN Y CONVERSIONES ==================
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

# ================== PARÁMETROS FÍSICOS ==================
# Masa (kg) de la "carga" 
m = 72.0
# Aceleración gravitatoria (m/s^2)
g = 9.81
# Longitud del péndulo (metros)
L = 2.32

# Factor de conversión temporal
time_per_frame = 1.0 / fps  # segundos por frame
print(f"\nFactor temporal: 1 frame = {time_per_frame:.4f} segundos")

# Limpiar datos: eliminar None y convertir a arrays
posiciones_limpias = {}
indices_validos = {}  # Guardar índices para reconstruir tiempo correcto
for color in posiciones:
    # Filtrar None y guardar índices válidos
    coords_validas = []
    indices = []
    for i, pos in enumerate(posiciones[color]):
        if pos is not None:
            coords_validas.append(pos)
            indices.append(i)
    
    if coords_validas:
        posiciones_limpias[color] = np.array(coords_validas)
        indices_validos[color] = np.array(indices)
        print(f"{color.capitalize()}: {len(coords_validas)}/{frame_count} detecciones ({100*len(coords_validas)/frame_count:.1f}%)")
    else:
        posiciones_limpias[color] = np.array([])
        indices_validos[color] = np.array([])
        print(f"{color.capitalize()}: 0 detecciones")

        posiciones_limpias[color] = np.array([])
        indices_validos[color] = np.array([])
        print(f"{color.capitalize()}: 0 detecciones")

# ================== CALIBRACIÓN DE ESCALA ==================
# Usar el punto de suspensión seleccionado manualmente
print(f"\n=== PUNTO DE SUSPENSIÓN ===")
print(f"Pivote seleccionado: ({x_pivot:.1f}, {y_pivot:.1f}) px")

# Para cada color, calcular radio promedio y factor de conversión
meters_per_pixel = {}
resultados_polares = {}

for color in ['celeste', 'rojo_oscuro']:
    if len(posiciones_limpias[color]) == 0:
        print(f"\nNo hay datos para {color}")
        resultados_polares[color] = None
        continue
    
    print(f"\n=== ANÁLISIS PARA {color.upper()} ===")
    
    # Posiciones en píxeles
    x_pos = posiciones_limpias[color][:, 0]
    y_pos = posiciones_limpias[color][:, 1]
    
    # Aplicar suavizado
    x_pos = smooth_data(x_pos, window_size=5)
    y_pos = smooth_data(y_pos, window_size=5)
    
    # Coordenadas relativas al pivote (invertir Y para física: Y+ hacia arriba)
    x_rel = x_pos - x_pivot
    y_rel = -(y_pos - y_pivot)  # Invertir Y
    
    # Calcular radio y ángulo
    r_pixels = np.sqrt(x_rel**2 + y_rel**2)
    theta = np.arctan2(x_rel, y_rel) - np.pi  # Ángulo desde vertical hacia abajo
    theta_unwrapped = np.unwrap(theta)
    
    # Calibración: convertir píxeles a metros
    radio_promedio_pixels = np.mean(r_pixels)
    meters_per_pixel[color] = L / radio_promedio_pixels
    print(f"Calibración: {meters_per_pixel[color]:.6f} m/px (radio promedio: {radio_promedio_pixels:.1f} px)")
    
    # Convertir radio a metros
    r_m = r_pixels * meters_per_pixel[color]
    
    # Tiempos basados en índices válidos
    time_pos = indices_validos[color] * time_per_frame
    
    # Velocidades polares
    vr_raw = np.diff(r_m) / time_per_frame  # Velocidad radial (m/s)
    vtheta_raw = np.diff(theta_unwrapped) / time_per_frame  # Velocidad angular (rad/s)
    
    # Suavizar velocidades
    vr = smooth_data(vr_raw, window_size=5)
    omega = smooth_data(vtheta_raw, window_size=5)
    
    # Tiempo para velocidades (un elemento menos por el diff)
    time_vel = time_pos[1:]
    
    # Guardar resultados
    resultados_polares[color] = {
        'time_pos': time_pos,
        'time_vel': time_vel,
        'r_m': r_m,
        'theta': theta_unwrapped,
        'vr': vr,
        'omega': omega,
        'x_pos': x_pos,
        'y_pos': y_pos
    }
    
    print(f"Radio: {np.mean(r_m):.3f} ± {np.std(r_m):.3f} m")
    print(f"Velocidad radial máxima: {np.max(np.abs(vr)):.3f} m/s")
    print(f"Velocidad angular máxima: {np.max(np.abs(omega)):.3f} rad/s = {np.degrees(np.max(np.abs(omega))):.1f}°/s")

# Graficar trayectorias
# Graficar trayectorias
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
if len(posiciones_limpias['celeste']) > 0:
    plt.plot(posiciones_limpias['celeste'][:, 0], posiciones_limpias['celeste'][:, 1], 'c-', linewidth=1, alpha=0.7, label='Parche Celeste')
    plt.scatter(posiciones_limpias['celeste'][0, 0], posiciones_limpias['celeste'][0, 1], c='green', s=100, marker='o', label='Inicio')
    plt.scatter(posiciones_limpias['celeste'][-1, 0], posiciones_limpias['celeste'][-1, 1], c='red', s=100, marker='x', label='Fin')
plt.title('Trayectoria del Parche Celeste')
plt.xlabel('X (píxeles)')
plt.ylabel('Y (píxeles)')
plt.legend()
plt.gca().invert_yaxis()
plt.grid(alpha=0.3)

plt.subplot(2, 2, 2)
if len(posiciones_limpias['rojo_oscuro']) > 0:
    plt.plot(posiciones_limpias['rojo_oscuro'][:, 0], posiciones_limpias['rojo_oscuro'][:, 1], color='darkred', linewidth=1, alpha=0.7, label='Hamaca (rojo oscuro)')
    plt.scatter(posiciones_limpias['rojo_oscuro'][0, 0], posiciones_limpias['rojo_oscuro'][0, 1], c='green', s=100, marker='o', label='Inicio')
    plt.scatter(posiciones_limpias['rojo_oscuro'][-1, 0], posiciones_limpias['rojo_oscuro'][-1, 1], c='red', s=100, marker='x', label='Fin')
plt.title('Trayectoria de la Hamaca (Rojo Oscuro)')
plt.xlabel('X (píxeles)')
plt.ylabel('Y (píxeles)')
plt.legend()
plt.gca().invert_yaxis()
plt.grid(alpha=0.3)

# Gráficos de posición X vs tiempo
plt.subplot(2, 2, 3)
if len(posiciones_limpias['celeste']) > 0:
    time_celeste = np.arange(len(posiciones_limpias['celeste'])) / fps
    plt.plot(time_celeste, posiciones_limpias['celeste'][:, 0], 'c-', label='Celeste X')
plt.title('Posición X del Parche Celeste vs Tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición X (píxeles)')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2, 2, 4)
if len(posiciones_limpias['rojo_oscuro']) > 0:
    time_rojo = np.arange(len(posiciones_limpias['rojo_oscuro'])) / fps
    plt.plot(time_rojo, posiciones_limpias['rojo_oscuro'][:, 0], color='darkred', label='Hamaca X')
plt.title('Posición X de la Hamaca vs Tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición X (píxeles)')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ================== GRÁFICOS DE VELOCIDADES POLARES ==================
if resultados_polares['celeste'] is not None or resultados_polares['rojo_oscuro'] is not None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: Velocidad angular del brazo (celeste)
    ax1 = axes[0, 0]
    if resultados_polares['celeste'] is not None:
        data_celeste = resultados_polares['celeste']
        ax1.plot(data_celeste['time_vel'], data_celeste['omega'], 'c-', linewidth=2, label='Velocidad angular (ω)')
        ax1.plot(data_celeste['time_vel'], data_celeste['vr'], 'b--', linewidth=1.5, alpha=0.7, label='Velocidad radial (vᵣ)')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Velocidad angular (rad/s) / Velocidad radial (m/s)')
        ax1.set_title('Velocidades Polares del Parche Celeste (Brazo)')
        ax1.legend()
        ax1.grid(alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Sin datos para parche celeste', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Velocidades Polares del Parche Celeste (Brazo)')
    
    # Gráfico 2: Velocidad angular de la hamaca (rojo oscuro)
    ax2 = axes[0, 1]
    if resultados_polares['rojo_oscuro'] is not None:
        data_rojo = resultados_polares['rojo_oscuro']
        ax2.plot(data_rojo['time_vel'], data_rojo['omega'], color='darkred', linewidth=2, label='Velocidad angular (ω)')
        ax2.plot(data_rojo['time_vel'], data_rojo['vr'], 'r--', linewidth=1.5, alpha=0.7, label='Velocidad radial (vᵣ)')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Velocidad angular (rad/s) / Velocidad radial (m/s)')
        ax2.set_title('Velocidades Polares de la Hamaca (Rojo Oscuro)')
        ax2.legend()
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Sin datos para parche rojo oscuro', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Velocidades Polares de la Hamaca (Rojo Oscuro)')
    
    # Gráfico 3: Comparación de velocidades angulares
    ax3 = axes[1, 0]
    if resultados_polares['celeste'] is not None:
        data_celeste = resultados_polares['celeste']
        ax3.plot(data_celeste['time_vel'], data_celeste['omega'], 'c-', linewidth=2, label='Brazo (celeste)')
    if resultados_polares['rojo_oscuro'] is not None:
        data_rojo = resultados_polares['rojo_oscuro']
        ax3.plot(data_rojo['time_vel'], data_rojo['omega'], color='darkred', linewidth=2, label='Hamaca (rojo)')
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Velocidad Angular ω (rad/s)')
    ax3.set_title('Comparación de Velocidades Angulares')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Gráfico 4: Comparación de velocidades radiales
    ax4 = axes[1, 1]
    if resultados_polares['celeste'] is not None:
        data_celeste = resultados_polares['celeste']
        ax4.plot(data_celeste['time_vel'], data_celeste['vr'], 'b-', linewidth=2, label='Brazo (celeste)')
    if resultados_polares['rojo_oscuro'] is not None:
        data_rojo = resultados_polares['rojo_oscuro']
        ax4.plot(data_rojo['time_vel'], data_rojo['vr'], 'r-', linewidth=2, label='Hamaca (rojo)')
    ax4.set_xlabel('Tiempo (s)')
    ax4.set_ylabel('Velocidad Radial vᵣ (m/s)')
    ax4.set_title('Comparación de Velocidades Radiales')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== ANÁLISIS DE VELOCIDADES POLARES ===")
    for color in ['celeste', 'rojo_oscuro']:
        if resultados_polares[color] is not None:
            data = resultados_polares[color]
            print(f"\n{color.capitalize()}:")
            print(f"  Velocidad angular media: {np.mean(np.abs(data['omega'])):.3f} rad/s")
            print(f"  Velocidad angular máxima: {np.max(np.abs(data['omega'])):.3f} rad/s = {np.degrees(np.max(np.abs(data['omega']))):.1f}°/s")
            print(f"  Velocidad radial media: {np.mean(np.abs(data['vr'])):.3f} m/s")
            print(f"  Velocidad radial máxima: {np.max(np.abs(data['vr'])):.3f} m/s")

print(f"\n=== RESUMEN FINAL ===")
print(f"Frames procesados: {frame_count}")
print(f"Duración del video: {frame_count/fps:.2f} segundos")
for color in posiciones_limpias:
    if len(posiciones_limpias[color]) > 0:
        print(f"\n{color.capitalize()}:")
        print(f"  Detecciones exitosas: {len(posiciones_limpias[color])}/{frame_count} ({100*len(posiciones_limpias[color])/frame_count:.1f}%)")
        print(f"  Rango X: {posiciones_limpias[color][:, 0].min():.0f} - {posiciones_limpias[color][:, 0].max():.0f} px")
        print(f"  Rango Y: {posiciones_limpias[color][:, 1].min():.0f} - {posiciones_limpias[color][:, 1].max():.0f} px")