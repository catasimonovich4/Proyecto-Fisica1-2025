import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

print(cv2.__version__)

# Abrir el video
cap = cv2.VideoCapture(r"./Videos/clemen-normal.mp4")

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
bbox = cv2.selectROI("Seleccione la hamaca", frame, False)
cv2.destroyWindow("Seleccione la hamaca")  # Cerrar ventana de selección

# Seleccionar el punto de suspensión de la hamaca
print("\nAhora seleccione el punto de suspensión de la hamaca (donde está colgada)")
origin_roi = cv2.selectROI("Seleccione punto de suspension", frame, False)
cv2.destroyWindow("Seleccione punto de suspension")

# Calcular el origen real (punto de suspensión)
x_origin = origin_roi[0] + origin_roi[2] / 2
y_origin = origin_roi[1] + origin_roi[3] / 2

print(f"Punto de suspension seleccionado: ({x_origin:.1f}, {y_origin:.1f})")

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
    # IMPORTANTE: Invertir Y para que crezca hacia arriba (física) en lugar de abajo (imagen)
    x_rel = x_pos - x_origin_final
    y_rel = -(y_pos - y_origin_final)  # Invertir eje Y

    # Calcular radio (pixeles) y ángulo respecto a la vertical (y hacia arriba positivo)
    r = np.sqrt(x_rel**2 + y_rel**2)
    # arctan2(x_rel, y_rel) da el ángulo desde +Y (vertical arriba)
    # Para física de péndulo, queremos ángulo desde -Y (vertical abajo)
    # Entonces restamos π para que θ=0 sea vertical hacia abajo
    theta_p = np.arctan2(x_rel, y_rel) - np.pi  # ángulo físico desde vertical hacia abajo
    theta_p_unwrapped = np.unwrap(theta_p)

    # ================== CALIBRACIÓN DE ESCALA ==================
    radio_real = 2.32  # metros
    radio_promedio_pixels = np.mean(r)
    meters_per_pixel = radio_real / radio_promedio_pixels
    pixels_per_meter = 1.0 / meters_per_pixel
    r_m = r * meters_per_pixel
    x_pos_m = x_pos * meters_per_pixel
    y_pos_m = y_pos * meters_per_pixel
    x_origin_m = x_origin_final * meters_per_pixel
    y_origin_m = y_origin_final * meters_per_pixel

    # Tiempos
    time_pos = np.arange(len(x_pos)) * time_per_frame
    time_vel = np.arange(1, len(x_pos)) * time_per_frame
    time_acc = np.arange(2, len(x_pos)) * time_per_frame

    # Velocidades cartesianas (pixeles -> m/s)
    vx_raw = np.diff(x_pos)
    vy_raw = np.diff(y_pos)
    vx = smooth_data(vx_raw, window_size=5)
    vy = smooth_data(vy_raw, window_size=5)
    vx_ms = vx * meters_per_pixel / time_per_frame
    vy_ms = vy * meters_per_pixel / time_per_frame

    # Aceleraciones cartesianas (m/s²)
    ax_raw = np.diff(vx)
    ay_raw = np.diff(vy)
    ax = smooth_data(ax_raw, window_size=5)
    ay = smooth_data(ay_raw, window_size=5)
    ax_ms2 = ax * meters_per_pixel / (time_per_frame**2)
    ay_ms2 = ay * meters_per_pixel / (time_per_frame**2)

    # ================== COORDENADAS POLARES ==================
    vr_raw = np.diff(r)
    vtheta_raw = np.diff(theta_p_unwrapped)
    vr = smooth_data(vr_raw, window_size=5)
    vtheta = smooth_data(vtheta_raw, window_size=5)
    vr_ms = vr * meters_per_pixel / time_per_frame
    omega = vtheta / time_per_frame  # rad/s
    theta_mid = theta_p_unwrapped[1:]  # alineado con omega

    # ================== FUERZAS (DINÁMICA) ==================
    L = np.mean(r_m)
    
    # DEBUG: Verificar valores de theta y componentes
    print(f"\n=== DEBUG ÁNGULOS ===")
    print(f"theta_mid mínimo: {np.min(theta_mid):.3f} rad = {np.degrees(np.min(theta_mid)):.1f}°")
    print(f"theta_mid máximo: {np.max(theta_mid):.3f} rad = {np.degrees(np.max(theta_mid)):.1f}°")
    print(f"cos(theta_mid) mínimo: {np.min(np.cos(theta_mid)):.3f}")
    print(f"cos(theta_mid) máximo: {np.max(np.cos(theta_mid)):.3f}")
    
    tension = m * g * np.cos(theta_mid) + m * L * omega**2
    tension = np.maximum(tension, 0.0)
    peso = m * g
    
    print(f"\n=== DEBUG FUERZAS ===")
    print(f"Tensión mínima: {np.min(tension):.1f} N")
    print(f"Tensión máxima: {np.max(tension):.1f} N")
    print(f"Tensión promedio: {np.mean(tension):.1f} N")
    print(f"Peso: {peso:.1f} N")
    print(f"Velocidad angular máxima: {np.max(np.abs(omega)):.3f} rad/s")
    print(f"Velocidad angular promedio: {np.mean(np.abs(omega)):.3f} rad/s")
    
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
    
    plt.subplot(3, 2, 3)
    plt.plot(time_pos, r_m, 'orange', label='Radio r')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Radio (m)')
    plt.title('Radio vs Tiempo (Coordenadas Polares)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.plot(time_pos, np.degrees(theta_p_unwrapped), 'purple', label='Ángulo θ')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Ángulo (grados)')
    plt.title('Ángulo vs Tiempo (Coordenadas Polares)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 5)
    plt.plot(time_vel, omega, 'purple', label='Velocidad angular')
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

    # ================== ANÁLISIS DE ENERGÍA ==================
    # Energía cinética: Ek = (1/2) * m * v^2
    E_cinetica = 0.5 * m * v_magnitude**2
    
    # Energía potencial gravitatoria: Ep = m * g * h
    # h es la altura respecto al punto más bajo de la trayectoria
    # Reconstruir posiciones Y desde ángulos para consistencia
    y_positions = r_m * (-np.cos(theta_p_unwrapped))  # Coordenadas Y físicas
    h = y_positions - np.min(y_positions)  # Altura respecto al mínimo
    E_potencial = m * g * h
    
    # Alinear tamaños: E_cinetica tiene 1 elemento menos por el diff en velocidades
    E_potencial_aligned = E_potencial[1:]  # Descartar primer elemento para alinear
    time_energia = time_vel  # Usar time_vel para energías alineadas
    
    # Energía mecánica total
    E_mecanica = E_cinetica + E_potencial_aligned
    
    # Crear figura para energías
    plt.figure(figsize=(10, 6), constrained_layout=True)
    
    plt.plot(time_energia, E_cinetica, 'r-', label='Energía Cinética', linewidth=1.5)
    plt.plot(time_energia, E_potencial_aligned, 'b-', label='Energía Potencial', linewidth=1.5)
    plt.plot(time_energia, E_mecanica, 'k-', label='Energía Mecánica Total', linewidth=2)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Energía (J)')
    plt.title('Energías vs Tiempo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()
    
    # Estadísticas de energía
    print(f"\n=== ANÁLISIS DE ENERGÍA ===")
    print(f"Energía mecánica inicial: {E_mecanica[0]:.2f} J")
    print(f"Energía mecánica final: {E_mecanica[-1]:.2f} J")
    print(f"Energía disipada: {E_mecanica[0] - E_mecanica[-1]:.2f} J")
    print(f"Porcentaje de energía disipada: {100 * (E_mecanica[0] - E_mecanica[-1]) / E_mecanica[0]:.1f}%")
    print(f"Energía cinética máxima: {np.max(E_cinetica):.2f} J")
    print(f"Energía potencial máxima: {np.max(E_potencial):.2f} J")

    # ================== DIAGRAMA DE CUERPO LIBRE ==================
    idx_max_v = np.argmax(np.abs(omega))
    th_fbd = theta_mid[idx_max_v]
    T_fbd = tension[idx_max_v]

    # Vector radial unitario desde pivote a masa:
    # theta=0 es vertical hacia abajo, crece antihorario
    # Entonces: r_hat debe apuntar hacia abajo cuando theta=0
    r_hat_fbd = np.array([np.sin(th_fbd), -np.cos(th_fbd)])  # Y negativo para colgar
    # Dirección tensión: desde masa hacia pivote (opuesto a r_hat)
    dir_T = -r_hat_fbd  # Apunta desde masa hacia pivote (hacia arriba)
    # Dirección peso: hacia abajo (Y negativo)
    dir_W = np.array([0.0, -1.0])

    # Escalar vectores para visualización
    T_max = np.max(tension)
    scale = 0.5 / T_max  # Factor de escala para que las flechas sean visibles
    vec_T = dir_T * T_fbd * scale
    vec_W = dir_W * peso * scale

    # Posición de la masa (bob)
    bob_pos = r_hat_fbd * L * 0.4

    plt.figure(figsize=(6,6))
    
    # Dibujar la cuerda desde el pivote hasta la masa
    plt.plot([0, bob_pos[0]], [0, bob_pos[1]], 'k-', linewidth=2, label='Cuerda')
    
    # Dibujar el pivote
    plt.scatter([0], [0], color='black', s=100, marker='o', zorder=5, label='Pivote')
    
    # Dibujar la masa
    plt.scatter([bob_pos[0]], [bob_pos[1]], color='red', s=200, zorder=5, label='Masa')
    
    # Dibujar las FUERZAS desde la posición de la masa
    plt.quiver(bob_pos[0], bob_pos[1], vec_T[0], vec_T[1], 
               angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.01, label=f'Tensión ({T_fbd:.1f} N)')
    plt.quiver(bob_pos[0], bob_pos[1], vec_W[0], vec_W[1], 
               angles='xy', scale_units='xy', scale=1, 
               color='green', width=0.01, label=f'Peso ({peso:.1f} N)')
    
    plt.title(f'Diagrama de Cuerpo Libre (ángulo = {np.degrees(th_fbd):.1f}°)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.xlim(-L*0.7, L*0.7)
    plt.ylim(-0.2, L*0.6)
    plt.show()

    # ================== FLECHAS EN TRAYECTORIA ==================
    step = max(1, len(theta_mid)//40)
    sample_indices = np.arange(0, len(theta_mid), step)
    
    # Reconstruir posiciones desde ángulos para consistencia con el diagrama de cuerpo libre
    # Usar los mismos vectores radiales que en el diagrama
    x_traj = []
    y_traj = []
    for j in range(len(theta_p_unwrapped)):
        th = theta_p_unwrapped[j]
        r_j = r_m[j]
        # Vector radial consistente: r_hat = [sin(theta), -cos(theta)]
        x_traj.append(r_j * np.sin(th))
        y_traj.append(r_j * (-np.cos(th)))
    
    x_traj = np.array(x_traj)
    y_traj = np.array(y_traj)
    x_rel_mid = x_traj[1:]
    y_rel_mid = y_traj[1:]

    plt.figure(figsize=(12,6), constrained_layout=True)
    ax1 = plt.subplot(1,2,1)
    ax1.plot(x_traj, y_traj, color='gray', linewidth=1, label='Trayectoria')

    for i in sample_indices:
        th = theta_mid[i]
        bx = x_rel_mid[i]; by = y_rel_mid[i]
        # Vector radial: theta=0 es vertical abajo, crece antihorario
        r_hat = np.array([np.sin(th), -np.cos(th)])  # Y negativo para colgar
        dir_T = -r_hat  # Tensión apunta desde masa hacia pivote (opuesto a r_hat)
        dir_W = np.array([0.0, -1.0])  # Peso hacia abajo (Y negativo)
        T_mag = tension[i] * scale * L * 0.6
        W_mag = peso * scale * L * 0.6
        ax1.quiver(bx, by, dir_T[0]*T_mag, dir_T[1]*T_mag, angles='xy', scale_units='xy', scale=1, color='blue')
        ax1.quiver(bx, by, dir_W[0]*W_mag, dir_W[1]*W_mag, angles='xy', scale_units='xy', scale=1, color='green')

    ax1.scatter([0],[0], color='black', s=30, label='Pivote')
    ax1.set_title('Trayectoria y Fuerzas'); ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.axis('equal'); ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = plt.subplot(1,2,2)
    ax2.plot(time_vel, tension, color='blue', label='Tensión (N)')
    ax2.axhline(peso, color='green', linestyle='--', label='Peso (N)')
    ax2.set_xlabel('Tiempo (s)'); ax2.set_ylabel('Fuerza (N)'); ax2.set_title('Fuerzas vs Tiempo')
    ax2.legend(); ax2.grid(alpha=0.3)
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
    print(f"Velocidad angular máxima: {np.max(np.abs(omega)):.3f} rad/s = {np.degrees(np.max(np.abs(omega))):.3f}°/s")
    
    # Análisis del período de oscilación
    peaks, _ = find_peaks(np.degrees(theta_p_unwrapped), height=None, distance=20)
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
