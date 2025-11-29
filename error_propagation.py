"""
Módulo de Propagación de Errores para Análisis de Movimiento Pendular
=====================================================================

Calcula y propaga incertidumbres en:
- Calibración (píxeles → metros)
- Posiciones y velocidades
- Magnitudes derivadas (aceleración, ángulo, fuerzas)

Basado en teoría de propagación de errores por derivadas parciales.
"""

import numpy as np
from typing import Tuple, Dict


class ErrorAnalysis:
    """Clase para análisis sistemático de errores en el experimento de tracking."""
    
    def __init__(self, fps: float, m: float = 72.0, g: float = 9.81):
        """
        Inicializa los parámetros físicos y precisiones del instrumento.
        
        Args:
            fps: Frames por segundo del video
            m: Masa del sistema (kg)
            g: Aceleración de gravedad (m/s²)
        """
        self.fps = fps
        self.m = m
        self.g = g
        self.dt = 1.0 / fps
        
        # ===== ERRORES DE MEDICIONES DIRECTAS =====
        # Error de apreciación de la cámara (± píxeles)
        self.pixel_error = 1.0  # ±1 píxel (resolución mínima)
        
        # Error temporal (±0.5 frames)
        self.time_error_frame = 0.5 / fps  # segundos
        
        # Error de instrumentos de calibración
        self.tape_measure_error = 0.01  # ±1 cm (cinta métrica)
        
        # Error en gravedad (referencia)
        self.g_error = 0.2  # ±0.2 m/s²
        
        # Error en masa (asumir preciso si fue pesada)
        self.m_error = 0.5  # ±0.5 kg
        
        # Guardamos resultados de calibración
        self.meters_per_pixel = None
        self.meters_per_pixel_error = None
    
    def calibrate_scale(self, 
                       radio_promedio_pixels: float,
                       radio_promedio_pixels_std: float,
                       radio_real_m: float = 2.32) -> Tuple[float, float]:
        """
        Calcula el factor de conversión píxeles → metros con su incertidumbre.
        
        Args:
            radio_promedio_pixels: Radio promedio medido en píxeles
            radio_promedio_pixels_std: Desviación estándar del radio en píxeles
            radio_real_m: Radio real en metros (medido con cinta)
        
        Returns:
            (meters_per_pixel, meters_per_pixel_error)
        
        Explicación:
            - Mediste el radio con cinta métrica: 2.32 m ± 0.01 m
            - En la imagen mide: 500 px ± 3 px (variabilidad del tracking)
            - Conversión: escala = radio_real / radio_px
            - Error: propaga errores de AMBAS mediciones
        """
        self.meters_per_pixel = radio_real_m / radio_promedio_pixels
        
        # Propagación por derivadas parciales para f = A/B
        # σ_f = f * √[(σ_A/A)² + (σ_B/B)²]
        rel_error_real = (self.tape_measure_error / radio_real_m) ** 2
        rel_error_pix = (radio_promedio_pixels_std / radio_promedio_pixels) ** 2
        
        self.meters_per_pixel_error = self.meters_per_pixel * np.sqrt(rel_error_real + rel_error_pix)
        
        return self.meters_per_pixel, self.meters_per_pixel_error
    
    def position_error(self, 
                      x_pos_m: np.ndarray,
                      y_pos_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula error en posiciones (metros) considerando:
        1. Error de tracking (±píxeles)
        2. Error de calibración
        
        Args:
            x_pos_m: Array de posiciones X en metros
            y_pos_m: Array de posiciones Y en metros
        
        Returns:
            (x_error_m, y_error_m) - Errores en metros
        
        Explicación:
            Cada coordenada tiene dos fuentes de error:
            a) Error de localización del tracker: ±√2 píxeles en cada dirección
            b) Incertidumbre del factor de calibración aplicado a la posición
            
            Error total = √[error_tracker² + error_calibración²]
        """
        # Error de tracking en píxeles (combina X e Y)
        pixel_error_total = self.pixel_error * np.sqrt(2)  # Teorema Pitágoras
        
        # Error por imprecisión de localización
        error_from_tracking = pixel_error_total * self.meters_per_pixel
        
        # Error por incertidumbre del factor de calibración
        # Si x_pos = 100 px y escala tiene error ±0.000015 m/px
        # Entonces error = 100 * 0.000015 = 0.0015 m
        error_from_calibration_x = np.abs(x_pos_m * self.meters_per_pixel_error / self.meters_per_pixel)
        error_from_calibration_y = np.abs(y_pos_m * self.meters_per_pixel_error / self.meters_per_pixel)
        
        # Combinación en cuadratura (errores independientes)
        x_error_m = np.sqrt(error_from_tracking**2 + error_from_calibration_x**2)
        y_error_m = np.sqrt(error_from_tracking**2 + error_from_calibration_y**2)
        
        return x_error_m, y_error_m
    
    def velocity_error(self,
                      v_x: np.ndarray,
                      v_y: np.ndarray,
                      pos_error_x: np.ndarray,
                      pos_error_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula error en velocidades.
        
        v = Δx / Δt
        
        Derivadas parciales:
            ∂v/∂x = 1/Δt  (si Δx aumenta, v aumenta)
            ∂v/∂t = -Δx/Δt²  (si Δt aumenta, v disminuye)
        
        Error: σ_v = √[(∂v/∂x·σ_x)² + (∂v/∂t·σ_t)²]
        """
        # Error por incertidumbre en posición
        error_from_position_x = pos_error_x / self.dt
        error_from_position_y = pos_error_y / self.dt
        
        # Error por incertidumbre temporal
        # Si v = 0.5 m/s y Δt tiene error de 0.01 s
        # Entonces error en v ≈ 0.5 * 0.01 / 0.01² ≈ 0.5 / 0.01 = 50 m/s (extremo)
        error_from_time_x = np.abs(v_x) * self.time_error_frame / self.dt
        error_from_time_y = np.abs(v_y) * self.time_error_frame / self.dt
        
        # Combinación en cuadratura
        v_x_error = np.sqrt(error_from_position_x**2 + error_from_time_x**2)
        v_y_error = np.sqrt(error_from_position_y**2 + error_from_time_y**2)
        
        return v_x_error, v_y_error
    
    def angle_error(self,
                    x_rel: np.ndarray,
                    y_rel: np.ndarray,
                    x_error: np.ndarray,
                    y_error: np.ndarray) -> np.ndarray:
        """
        Calcula error en ángulo θ = arctan2(x_rel, y_rel)
        
        Derivadas parciales de arctan2(x, y):
            ∂θ/∂x = y / (x² + y²)
            ∂θ/∂y = -x / (x² + y²)
        
        Error: σ_θ = √[(∂θ/∂x·σ_x)² + (∂θ/∂y·σ_y)²]
        """
        r_sq = x_rel**2 + y_rel**2
        r_sq = np.maximum(r_sq, 1e-6)  # Evitar división por cero
        
        # Derivadas parciales
        dtheta_dx = y_rel / r_sq
        dtheta_dy = -x_rel / r_sq
        
        # Propagación
        theta_error = np.sqrt((dtheta_dx * x_error)**2 + (dtheta_dy * y_error)**2)
        
        return theta_error
    
    def angular_velocity_error(self,
                               theta: np.ndarray,
                               theta_error: np.ndarray) -> np.ndarray:
        """
        Calcula error en velocidad angular ω = dθ/dt
        
        Similar a velocidad lineal:
            ∂ω/∂θ = 1/Δt
            ∂ω/∂t = -dθ/Δt²
        """
        # Error por incertidumbre angular
        error_from_angle = theta_error / self.dt
        
        # Error temporal: para ω es menos crítico que para v
        # porque los ángulos varían más lentamente
        dtheta = np.diff(theta)
        dtheta = np.concatenate([[dtheta[0]], dtheta])  # Alinear tamaño
        error_from_time = np.abs(dtheta / self.dt) * self.time_error_frame / self.dt
        
        omega_error = np.sqrt(error_from_angle[:-1]**2 + error_from_time[:-1]**2)
        
        return omega_error
    
    def tension_error(self,
                      theta: np.ndarray,
                      omega: np.ndarray,
                      L: float,
                      theta_error: np.ndarray,
                      omega_error: np.ndarray,
                      L_error: float) -> np.ndarray:
        """
        Calcula error en tensión T = m·g·cos(θ) + m·L·ω²
        
        Derivadas parciales:
            ∂T/∂m = g·cos(θ) + L·ω²
            ∂T/∂g = m·cos(θ)
            ∂T/∂θ = -m·g·sin(θ)
            ∂T/∂L = m·ω²
            ∂T/∂ω = 2·m·L·ω
        
        Explicación de fuentes de error:
        1. Error en masa (si pesaste) → muy pequeño
        2. Error en gravedad (0.2 m/s²) → pequeño pero sistemático
        3. Error en ángulo → afecta el cos(θ)
        4. Error en L (variabilidad del radio) → pequeño
        5. Error en ω → DOMINANTE (el ω² amplifica mucho el error)
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Derivadas parciales
        dT_dm = self.g * cos_theta + L * omega**2
        dT_dg = self.m * cos_theta
        dT_dtheta = -self.m * self.g * sin_theta
        dT_dL = self.m * omega**2
        dT_domega = 2 * self.m * L * omega
        
        # Propagación en cuadratura
        tension_error = np.sqrt(
            (dT_dm * self.m_error)**2 +
            (dT_dg * self.g_error)**2 +
            (dT_dtheta * theta_error)**2 +
            (dT_dL * L_error)**2 +
            (dT_domega * omega_error)**2
        )
        
        return tension_error
    
    def energy_error(self,
                     kinetic_energy: np.ndarray,
                     potential_energy: np.ndarray,
                     v_error: np.ndarray,
                     h_error: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula error en energía cinética y potencial.
        
        E_k = (1/2) * m * v²  →  ∂E_k/∂v = m·v
        E_p = m * g * h  →  ∂E_p/∂h = m·g
        """
        # Alinear tamaños
        min_len = min(len(kinetic_energy), len(v_error))
        v_error = v_error[:min_len]
        
        # Error energía cinética
        # E_k = (1/2) * m * v²  → dE_k = m * v * dv
        Ek_error = self.m * np.abs(kinetic_energy[:min_len]) * v_error / (0.5 * self.m + 1e-10)
        Ek_error = np.minimum(Ek_error, kinetic_energy[:min_len])  # Cap al valor mismo
        
        # Error energía potencial
        Ep_error = self.m * self.g * h_error
        
        # Error energía total
        E_total_error = np.sqrt(Ek_error**2 + Ep_error**2)
        
        return Ek_error, Ep_error, E_total_error
    
    def format_value_with_error(self, value: float, error: float, sig_figs: int = 1) -> str:
        """
        Formatea un valor con su error con una cifra significativa en el error.
        
        Ejemplo: format_value_with_error(1.2345, 0.0678) → "1.23 ± 0.07"
        """
        if error == 0 or np.isnan(error):
            return f"{value:.3f} ± 0.000"
        
        # Orden de magnitud del error
        magnitude = 10 ** np.floor(np.log10(np.abs(error)))
        error_rounded = np.round(error / magnitude, sig_figs) * magnitude
        
        # Decimal places basado en el error
        decimal_places = max(0, -int(np.floor(np.log10(error_rounded))))
        value_rounded = np.round(value, decimal_places)
        
        return f"[{value_rounded:.{decimal_places}f} ± {error_rounded:.{decimal_places}f}]"
    
    def print_summary(self,
                     measurements: Dict[str, Tuple[float, float]]) -> None:
        """
        Imprime un resumen de todas las mediciones con sus errores.
        
        Args:
            measurements: Dict con {nombre: (valor, error), ...}
        """
        print("\n" + "="*80)
        print("RESUMEN DE ANÁLISIS DE ERRORES - MOVIMIENTO PENDULAR")
        print("="*80)
        print(f"{'Magnitud':<30} {'Valor con Error':<25} {'Error %':<15}")
        print("-"*80)
        
        for name, (value, error) in measurements.items():
            if error > 0:
                error_percent = (error / abs(value)) * 100 if value != 0 else 0
                formatted = self.format_value_with_error(value, error)
                print(f"{name:<30} {formatted:<25} {error_percent:>6.2f}%")
            else:
                print(f"{name:<30} {value:<25} {'N/A':>6}")
        
        print("="*80 + "\n")


# Función auxiliar para crear análisis rápido
def quick_analysis(fps: float, 
                  r_pixels_mean: float,
                  r_pixels_std: float,
                  r_real_m: float = 2.32) -> ErrorAnalysis:
    """
    Crea un análisis de errores con parámetros típicos.
    
    Uso:
        error_analysis = quick_analysis(fps=30, r_pixels_mean=500, r_pixels_std=3)
        meters_per_pixel, error = error_analysis.calibrate_scale(r_pixels_mean, r_pixels_std)
    """
    error_analysis = ErrorAnalysis(fps=fps)
    error_analysis.calibrate_scale(r_pixels_mean, r_pixels_std, r_real_m)
    return error_analysis
