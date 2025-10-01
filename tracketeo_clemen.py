import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

print(cv2.__version__)

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

# Seleccionar el punto de suspensión de la hamaca
print("Ahora selecciona el punto de suspensión de la hamaca (donde está colgada)")
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

# Procesar las posiciones para graficar
if len(positions) > 1:
    pos_array = np.array(positions)
    x_pos = pos_array[:, 0]
    y_pos = pos_array[:, 1]

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

    # Calcular velocidad (diferencias entre posiciones)
    vx = np.diff(x_pos)
    vy = np.diff(y_pos)
    
    # Calcular aceleración (segunda derivada)
    ax = np.diff(vx)
    ay = np.diff(vy)

    frames_pos = np.arange(len(x_pos))         # Para posiciones
    frames_vel = np.arange(1, len(x_pos))      # Para velocidades
    frames_acc = np.arange(2, len(x_pos))      # Para aceleraciones

    # ================== COORDENADAS CARTESIANAS ==================
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 3, 1)
    plt.plot(frames_pos, x_pos, label='Posición X', color='b')
    plt.plot(frames_pos, y_pos, label='Posición Y', color='g')
    plt.xlabel('Frame')
    plt.ylabel('Posición (pixeles)')
    plt.title('Posición Cartesiana vs Tiempo')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 2)
    plt.plot(frames_vel, vx, label='Velocidad en X', color='r')
    plt.plot(frames_vel, vy, label='Velocidad en Y', color='m')
    plt.xlabel('Frame')
    plt.ylabel('Velocidad (pixeles/frame)')
    plt.title('Velocidad Cartesiana vs Tiempo')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 3)
    plt.plot(x_pos, y_pos, 'b-', linewidth=2, label='Trayectoria')
    plt.plot(x_origin_final, y_origin_final, 'ro', markersize=8, label='Origen final')
    
    # Mostrar líneas desde el origen a algunos puntos para verificar visualmente
    for i in range(0, len(x_pos), 50):  # Cada 50 frames
        plt.plot([x_origin_final, x_pos[i]], [y_origin_final, y_pos[i]], 'r--', alpha=0.3)
    
    # Mostrar círculo del radio promedio
    circle_r = np.mean(r)
    circle = plt.Circle((x_origin_final, y_origin_final), circle_r, fill=False, 
                       color='green', linestyle='--', alpha=0.5, label=f'Radio promedio: {circle_r:.0f}px')
    plt.gca().add_patch(circle)
    
    plt.xlabel('X (pixeles)')
    plt.ylabel('Y (pixeles)')
    plt.title('Trayectoria en el Plano XY')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # ================== COORDENADAS POLARES ==================
    # Debug: Análisis de los ángulos
    print(f"\n=== DEBUG COORDENADAS POLARES ===")
    print(f"Radio promedio: {np.mean(r):.1f} ± {np.std(r):.1f} pixeles")
    print(f"Rango de ángulos: {np.degrees(np.min(theta)):.1f}° a {np.degrees(np.max(theta)):.1f}°")
    print(f"Variación angular total: {np.degrees(np.max(theta) - np.min(theta)):.1f}°")
    
    # Manejar discontinuidades en theta (salto de -π a π)
    theta_unwrapped = np.unwrap(theta)
    
    # Velocidades polares
    vr = np.diff(r)
    vtheta = np.diff(theta_unwrapped)
    
    print(f"Velocidad angular máxima: {np.max(np.abs(vtheta)):.6f} rad/frame")
    print(f"Velocidad angular promedio: {np.mean(np.abs(vtheta)):.6f} rad/frame")
    
    plt.subplot(3, 3, 4)
    plt.plot(frames_pos, r, 'purple', label='Radio r')
    plt.xlabel('Frame')
    plt.ylabel('Radio (pixeles)')
    plt.title('Radio vs Tiempo (Coordenadas Polares)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 5)
    plt.plot(frames_pos, np.degrees(theta_unwrapped), 'orange', label='Ángulo θ')
    plt.xlabel('Frame')
    plt.ylabel('Ángulo (grados)')
    plt.title('Ángulo vs Tiempo (Coordenadas Polares)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 6)
    plt.plot(frames_vel, vr, 'purple', label='Velocidad radial')
    plt.plot(frames_vel, vtheta, 'orange', label='Velocidad angular')
    plt.xlabel('Frame')
    plt.ylabel('Velocidad')
    plt.title('Velocidades Polares vs Tiempo')
    plt.legend()
    plt.grid(True)

    # ================== COORDENADAS INTRÍNSECAS ==================
    # Calcular velocidad escalar y aceleración
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Vector tangente unitario (dirección de la velocidad)
    v_mag_safe = np.where(v_magnitude > 1e-6, v_magnitude, 1e-6)  # Evitar división por cero
    t_x = vx / v_mag_safe
    t_y = vy / v_mag_safe
    
    # Vector normal unitario (perpendicular al tangente, hacia la izquierda)
    n_x = -t_y
    n_y = t_x
    
    # Componentes intrínsecas de la aceleración
    if len(ax) > 0 and len(ay) > 0:
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
        a_tangential = ax * t_x_acc + ay * t_y_acc
        
        # Aceleración normal (componente perpendicular al movimiento)
        a_normal = ax * n_x_acc + ay * n_y_acc
        
        # Radio de curvatura
        v_mag_curvature = v_magnitude[1:]  # Velocidad en los puntos donde tenemos aceleración
        radius_curvature = np.where(np.abs(a_normal) > 1e-6, 
                                  v_mag_curvature**2 / np.abs(a_normal), 
                                  np.inf)
    else:
        a_tangential = np.array([])
        a_normal = np.array([])
        radius_curvature = np.array([])
    
    plt.subplot(3, 3, 7)
    plt.plot(frames_vel, v_magnitude, 'cyan', label='Velocidad escalar')
    plt.xlabel('Frame')
    plt.ylabel('Velocidad (pixeles/frame)')
    plt.title('Velocidad Escalar vs Tiempo')
    plt.legend()
    plt.grid(True)
    
    if len(a_tangential) > 0:
        plt.subplot(3, 3, 8)
        plt.plot(frames_acc, a_tangential, 'red', label='Aceleración tangencial')
        plt.plot(frames_acc, a_normal, 'blue', label='Aceleración normal')
        plt.xlabel('Frame')
        plt.ylabel('Aceleración (pixeles/frame²)')
        plt.title('Aceleraciones Intrínsecas vs Tiempo')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 3, 9)
        # Filtrar valores infinitos para el gráfico
        radius_finite = np.where(np.isfinite(radius_curvature), radius_curvature, np.nan)
        plt.plot(frames_acc, radius_finite, 'green', label='Radio de curvatura')
        plt.xlabel('Frame')
        plt.ylabel('Radio (pixeles)')
        plt.title('Radio de Curvatura vs Tiempo')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Escala logarítmica para mejor visualización

    plt.tight_layout()
    plt.show()
    
    # ================== RESUMEN DE DATOS ==================
    print("\n=== RESUMEN DEL ANÁLISIS ===")
    print(f"Número total de frames analizados: {len(positions)}")
    print(f"Origen final utilizado: ({x_origin_final:.1f}, {y_origin_final:.1f})")
    print(f"Radio promedio: {np.mean(r):.1f} ± {np.std(r):.1f} pixeles")
    print(f"Radio mínimo: {np.min(r):.1f} pixeles")
    print(f"Radio máximo: {np.max(r):.1f} pixeles")
    print(f"Variación del radio: {((np.max(r) - np.min(r)) / np.mean(r) * 100):.1f}%")
    print(f"Velocidad escalar promedio: {np.mean(v_magnitude):.2f} pixeles/frame")
    print(f"Velocidad angular máxima: {np.max(np.abs(vtheta)):.6f} rad/frame = {np.degrees(np.max(np.abs(vtheta))):.3f}°/frame")
    
    # Análisis del período de oscilación
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(np.degrees(theta_unwrapped), height=None, distance=20)
    if len(peaks) > 1:
        periodo_frames = np.mean(np.diff(peaks)) * 2  # *2 porque son semi-oscilaciones
        print(f"Período estimado: {periodo_frames:.1f} frames")
    
    if len(radius_curvature) > 0:
        radius_finite_values = radius_curvature[np.isfinite(radius_curvature)]
        if len(radius_finite_values) > 0:
            print(f"Radio de curvatura promedio: {np.mean(radius_finite_values):.1f} pixeles")

else:
    print("No se detectaron suficientes posiciones para graficar.")
