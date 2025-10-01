import cv2
import numpy as np
import matplotlib.pyplot as plt

print(cv2.__version__)

# Funciones para cambio de coordenadas y análisis
def cartesian_to_polar(x_coords, y_coords, origin_x=None, origin_y=None):
    """
    Convierte coordenadas cartesianas a polares
    x_coords, y_coords: arrays con las coordenadas cartesianas
    origin_x, origin_y: origen del sistema polar (por defecto usa el primer punto)
    """
    if origin_x is None:
        origin_x = x_coords[0]
    if origin_y is None:
        origin_y = y_coords[0]
    
    # Trasladar al origen
    x_rel = x_coords - origin_x
    y_rel = y_coords - origin_y
    
    r = np.sqrt(x_rel**2 + y_rel**2)
    theta = np.arctan2(y_rel, x_rel)
    
    return r, theta

def calculate_velocity_polar(r, theta, dt=1):
    """
    Calcula velocidades en coordenadas polares
    r, theta: arrays con coordenadas polares
    dt: intervalo de tiempo (por defecto 1 frame)
    """
    dr_dt = np.diff(r) / dt
    dtheta_dt = np.diff(theta) / dt
    
    # Velocidad radial y tangencial
    v_r = dr_dt
    v_theta = r[1:] * dtheta_dt  # r * dθ/dt
    
    return v_r, v_theta

def calculate_intrinsic_components(x_pos, y_pos, dt=1):
    """
    Calcula componentes tangencial y normal de velocidad y aceleración
    """
    # Calcular velocidad cartesiana
    vx = np.diff(x_pos) / dt
    vy = np.diff(y_pos) / dt
    
    # Magnitud de velocidad
    v_mag = np.sqrt(vx**2 + vy**2)
    
    # Evitar división por cero
    v_mag_safe = np.where(v_mag == 0, 1e-10, v_mag)
    
    # Vector tangente unitario
    t_x = vx / v_mag_safe
    t_y = vy / v_mag_safe
    
    # Calcular aceleración cartesiana
    ax = np.diff(vx) / dt
    ay = np.diff(vy) / dt
    
    # Corregir indexación para que coincidan las dimensiones
    # ax, ay tienen longitud len(x_pos)-2
    # t_x, t_y tienen longitud len(x_pos)-1
    # Necesitamos que coincidan para el producto escalar
    
    # Usar los vectores tangentes en los puntos correctos
    t_x_for_accel = t_x[:-1]  # Eliminar el último elemento
    t_y_for_accel = t_y[:-1]  # Eliminar el último elemento
    
    # Aceleración tangencial (proyección de a sobre t)
    a_t = ax * t_x_for_accel + ay * t_y_for_accel
    
    # Vector normal unitario (perpendicular a tangente)
    n_x = -t_y_for_accel
    n_y = t_x_for_accel
    
    # Aceleración normal (proyección de a sobre n)
    a_n = ax * n_x + ay * n_y
    
    return v_mag, a_t, a_n

def smooth_angle(theta):
    """
    Suaviza los ángulos para evitar saltos de -π a π
    """
    theta_smooth = np.copy(theta)
    for i in range(1, len(theta)):
        diff = theta[i] - theta[i-1]
        if diff > np.pi:
            theta_smooth[i:] -= 2*np.pi
        elif diff < -np.pi:
            theta_smooth[i:] += 2*np.pi
    return theta_smooth

# Abrir el video
cap = cv2.VideoCapture(r"./Videos/clemen-solo.mp4")

# Leer el primer frame
ret, frame = cap.read()
if not ret:
    print("No se pudo leer el video")
    exit()

# Seleccionar el objeto (hamaca) con el mouse
bbox = cv2.selectROI("Selecciona la hamaca", frame, False)
cv2.destroyWindow("Selecciona la hamaca")  # Cerrar ventana de selección

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

# Procesar las posiciones para graficar
# ...existing code...

if len(positions) > 1:
    pos_array = np.array(positions)
    x_pos = pos_array[:, 0]
    y_pos = pos_array[:, 1]

    # Calcular velocidad cartesiana (diferencias entre posiciones)
    vx = np.diff(x_pos)
    vy = np.diff(y_pos)

    frames_pos = np.arange(len(x_pos))         # Para posiciones
    frames_vel = np.arange(1, len(x_pos))      # Para velocidades

    # ===== COORDENADAS POLARES =====
    # Convertir a coordenadas polares (usando el primer punto como origen)
    r, theta = cartesian_to_polar(x_pos, y_pos)
    theta_smooth = smooth_angle(theta)  # Suavizar ángulos
    
    # Calcular velocidades polares
    v_r, v_theta = calculate_velocity_polar(r, theta_smooth)
    
    # ===== COORDENADAS INTRÍNSECAS =====
    # Calcular componentes tangencial y normal
    v_mag, a_t, a_n = calculate_intrinsic_components(x_pos, y_pos)

    # ===== GRÁFICOS =====
    
    # 1. Gráficos cartesianos (originales)
    plt.figure(figsize=(15, 12))
    
    # Posición cartesiana
    plt.subplot(3, 3, 1)
    plt.plot(frames_pos, x_pos, label='Posición X', color='b')
    plt.plot(frames_pos, y_pos, label='Posición Y', color='g')
    plt.xlabel('Frame')
    plt.ylabel('Posición (pixeles)')
    plt.title('Posición Cartesiana vs Tiempo')
    plt.legend()
    plt.grid(True)

    # Velocidad cartesiana
    plt.subplot(3, 3, 2)
    plt.plot(frames_vel, vx, label='Velocidad en X', color='r')
    plt.plot(frames_vel, vy, label='Velocidad en Y', color='m')
    plt.xlabel('Frame')
    plt.ylabel('Velocidad (pixeles/frame)')
    plt.title('Velocidad Cartesiana vs Tiempo')
    plt.legend()
    plt.grid(True)

    # Trayectoria cartesiana
    plt.subplot(3, 3, 3)
    plt.plot(x_pos, y_pos, 'b-', alpha=0.7)
    plt.scatter(x_pos[0], y_pos[0], color='green', s=100, label='Inicio')
    plt.scatter(x_pos[-1], y_pos[-1], color='red', s=100, label='Final')
    plt.xlabel('X (pixeles)')
    plt.ylabel('Y (pixeles)')
    plt.title('Trayectoria Cartesiana')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # 2. Gráficos polares
    # Posición polar
    plt.subplot(3, 3, 4)
    plt.plot(frames_pos, r, label='Radio r', color='orange')
    plt.xlabel('Frame')
    plt.ylabel('Radio (pixeles)')
    plt.title('Radio vs Tiempo')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 5)
    plt.plot(frames_pos, theta_smooth * 180/np.pi, label='Ángulo θ', color='purple')
    plt.xlabel('Frame')
    plt.ylabel('Ángulo (grados)')
    plt.title('Ángulo vs Tiempo')
    plt.legend()
    plt.grid(True)

    # Velocidad polar
    plt.subplot(3, 3, 6)
    plt.plot(frames_vel, v_r, label='Velocidad radial', color='orange')
    plt.plot(frames_vel, v_theta, label='Velocidad tangencial', color='purple')
    plt.xlabel('Frame')
    plt.ylabel('Velocidad (pixeles/frame)')
    plt.title('Velocidades Polares vs Tiempo')
    plt.legend()
    plt.grid(True)

    # 3. Gráficos intrínsecos
    # Velocidad escalar
    plt.subplot(3, 3, 7)
    plt.plot(frames_vel, v_mag, label='|v|', color='darkblue', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Velocidad (pixeles/frame)')
    plt.title('Magnitud de Velocidad vs Tiempo')
    plt.legend()
    plt.grid(True)

    # Aceleraciones intrínsecas
    plt.subplot(3, 3, 8)
    frames_acc = np.arange(2, len(x_pos))  # Para aceleraciones (longitud len(x_pos)-2)
    plt.plot(frames_acc, a_t, label='Aceleración tangencial', color='darkred')
    plt.plot(frames_acc, a_n, label='Aceleración normal', color='darkgreen')
    plt.xlabel('Frame')
    plt.ylabel('Aceleración (pixeles/frame²)')
    plt.title('Aceleraciones Intrínsecas vs Tiempo')
    plt.legend()
    plt.grid(True)

    # Gráfico polar de la trayectoria
    plt.subplot(3, 3, 9, polar=True)
    plt.plot(theta_smooth, r, 'b-', alpha=0.7)
    plt.scatter(theta_smooth[0], r[0], color='green', s=100, label='Inicio')
    plt.scatter(theta_smooth[-1], r[-1], color='red', s=100, label='Final')
    plt.title('Trayectoria en Coordenadas Polares')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ===== INFORMACIÓN ADICIONAL =====
    print(f"\n=== ANÁLISIS DEL MOVIMIENTO ===")
    print(f"Número total de frames analizados: {len(positions)}")
    print(f"Posición inicial: ({x_pos[0]:.1f}, {y_pos[0]:.1f}) pixeles")
    print(f"Posición final: ({x_pos[-1]:.1f}, {y_pos[-1]:.1f}) pixeles")
    print(f"Distancia total recorrida: {np.sum(np.sqrt(vx**2 + vy**2)):.1f} pixeles")
    print(f"Velocidad promedio: {np.mean(v_mag):.2f} pixeles/frame")
    print(f"Velocidad máxima: {np.max(v_mag):.2f} pixeles/frame")
    
    # Análisis polar
    print(f"\n=== ANÁLISIS POLAR ===")
    print(f"Radio mínimo: {np.min(r):.1f} pixeles")
    print(f"Radio máximo: {np.max(r):.1f} pixeles")
    print(f"Variación angular total: {(theta_smooth[-1] - theta_smooth[0]) * 180/np.pi:.1f} grados")

else:
    print("No se detectaron suficientes posiciones para graficar.")
