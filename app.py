import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import Any

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


AVAILABLE_METHODS = ("polyfit", "least_squares", "gradient_descent")



# ============================================================
# PARSEO Y CONFIGURACION DE COORDENADAS
# ============================================================

def parse_point(text: str) -> tuple[float, float] | None:
    text = text.strip()
    if not text:
        return None

    values = [value for value in re.split(r"[,\s;]+", text) if value]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("El punto debe tener formato x,y. Ejemplo: 520,780")

    return float(values[0]), float(values[1])


def parse_roi(text: str) -> tuple[int, int, int, int] | None:
    text = text.strip()
    if not text:
        return None

    values = [value for value in re.split(r"[,\s;]+", text) if value]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("El ROI debe tener formato x1,y1,x2,y2. Ejemplo: 540,180,1060,700")

    x1, y1, x2, y2 = [int(round(float(value))) for value in values]
    return x1, y1, x2, y2


def normalize_roi(
    roi: tuple[int, int, int, int] | None,
    frame_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]

    if roi is None:
        return 0, 0, width, height

    x1, y1, x2, y2 = roi

    x_min = max(0, min(x1, x2))
    x_max = min(width, max(x1, x2))
    y_min = max(0, min(y1, y2))
    y_max = min(height, max(y1, y2))

    if x_max <= x_min or y_max <= y_min:
        return 0, 0, width, height

    return x_min, y_min, x_max, y_max


@dataclass
class CoordinateMapper:
    """
    Convierte puntos de la camara a coordenadas reales de la CNC.

    Convencion por defecto:
    - Origen real: rectangulo verde = X0 Y0.
    - Derecha en imagen = X negativo.
    - Arriba en imagen = Y negativo, porque en OpenCV el eje Y crece hacia abajo.
    - Z positivo es hacia arriba, compatible con GRBL.
    """

    origin_px: np.ndarray
    mm_per_pixel_x: float
    mm_per_pixel_y: float
    x_right_negative: bool = True
    y_up_negative: bool = True

    def pixel_to_machine(self, points_px: np.ndarray) -> np.ndarray:
        if len(points_px) == 0:
            return np.empty((0, 2), dtype=np.float64)

        points_px = points_px.astype(np.float64)
        delta = points_px - self.origin_px

        x_sign = -1.0 if self.x_right_negative else 1.0
        y_sign = 1.0 if self.y_up_negative else -1.0

        x_mm = x_sign * delta[:, 0] * self.mm_per_pixel_x
        y_mm = y_sign * delta[:, 1] * self.mm_per_pixel_y

        return np.column_stack((x_mm, y_mm))

    def machine_to_pixel(self, points_mm: np.ndarray) -> np.ndarray:
        if len(points_mm) == 0:
            return np.empty((0, 2), dtype=np.float64)

        points_mm = points_mm.astype(np.float64)

        x_sign = -1.0 if self.x_right_negative else 1.0
        y_sign = 1.0 if self.y_up_negative else -1.0

        px = self.origin_px[0] + points_mm[:, 0] / (x_sign * self.mm_per_pixel_x)
        py = self.origin_px[1] + points_mm[:, 1] / (y_sign * self.mm_per_pixel_y)

        return np.column_stack((px, py))


@dataclass
class RuntimeSelection:
    origin_px: np.ndarray | None = None
    roi: tuple[int, int, int, int] | None = None
    mode: str | None = None
    roi_first_corner: tuple[int, int] | None = None

    # Datos usados solo por la interfaz  para separar visualmente
    # la vista de camara y el panel derecho sin dañar los clics.
    visual_interface: int = 1
    interface_2_view_box: tuple[int, int, int, int] | None = None
    interface_2_source_shape: tuple[int, int] | None = None


def build_mapper(
    origin_px: np.ndarray | None,
    mm_per_pixel_x: float,
    mm_per_pixel_y: float,
    x_positive_right: bool,
    y_down_negative: bool,
) -> CoordinateMapper | None:
    if origin_px is None:
        return None

    return CoordinateMapper(
        origin_px=origin_px.astype(np.float64),
        mm_per_pixel_x=max(1e-9, mm_per_pixel_x),
        mm_per_pixel_y=max(1e-9, mm_per_pixel_y),
        x_right_negative=not x_positive_right,
        y_up_negative=not y_down_negative,
    )



def get_interface_2_auto_origin(
    roi: tuple[int, int, int, int] | None,
    frame_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Devuelve el origen automatico de la interfaz .

    El punto 0,0 se ubica en la esquina inferior izquierda del ROI.
    Si aun no existe un ROI, se usa la esquina inferior izquierda
    del frame completo de la camara.
    """
    x1, y1, _, y2 = normalize_roi(roi, frame_shape)

    origin_x = x1
    origin_y = max(y1, y2 - 1)

    return np.array([origin_x, origin_y], dtype=np.float64)

# ============================================================
# DETECCION DE PUNTOS ROJOS
# ============================================================

def detect_red_points(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """
    Detecta los puntos rojos solamente dentro del area de enfoque de la camara.
    El ROI corresponde al cuadrado blanco de la imagen.
    """
    x1, y1, x2, y2 = normalize_roi(roi, frame.shape)
    crop = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 80, 50], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([170, 80, 50], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8 or area > 1500:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        x = int(moments["m10"] / moments["m00"]) + x1
        y = int(moments["m01"] / moments["m00"]) + y1
        points.append((x, y))

    if not points:
        return np.empty((0, 2), dtype=np.int32)

    points_array = np.array(points, dtype=np.int32)
    points_array = points_array[np.argsort(points_array[:, 0])]
    return points_array


def select_regression_points(
    points_px: np.ndarray,
    mapper: CoordinateMapper,
    only_sector_3: bool,
    sector_tolerance_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convierte puntos de pixeles a coordenadas reales y filtra el sector III:
    X < 0, Y < 0.
    """
    points_mm_all = mapper.pixel_to_machine(points_px)

    if len(points_mm_all) == 0:
        return points_px, points_mm_all, np.zeros((0,), dtype=bool)

    finite_mask = np.isfinite(points_mm_all[:, 0]) & np.isfinite(points_mm_all[:, 1])

    if only_sector_3:
        tol = abs(sector_tolerance_mm)
        sector_mask = (points_mm_all[:, 0] < -tol) & (points_mm_all[:, 1] < -tol)
        mask = finite_mask & sector_mask
    else:
        mask = finite_mask

    return points_px[mask], points_mm_all[mask], mask


# ============================================================
# REGRESION LINEAL EN COORDENADAS REALES DE LA CNC
# ============================================================

def linear_regression(points_mm: np.ndarray) -> tuple[float, float] | None:
    if len(points_mm) < 2:
        return None

    x = points_mm[:, 0].astype(np.float64)
    y = points_mm[:, 1].astype(np.float64)

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    if len(x) < 2:
        return None

    unique_x = np.unique(x)
    if len(unique_x) < 2:
        return None

    try:
        slope, intercept = np.polyfit(x, y, 1)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None

    return float(slope), float(intercept)


def least_squares_regression(points_mm: np.ndarray) -> tuple[float, float] | None:
    if len(points_mm) < 2:
        return None

    x = points_mm[:, 0].astype(np.float64)
    y = points_mm[:, 1].astype(np.float64)

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    if len(x) < 2:
        return None

    unique_x = np.unique(x)
    if len(unique_x) < 2:
        return None

    design = np.column_stack((x, np.ones_like(x)))
    try:
        solution, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    slope = float(solution[0])
    intercept = float(solution[1])

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None

    return slope, intercept


def gradient_descent_regression(points_mm: np.ndarray) -> tuple[float, float] | None:
    if len(points_mm) < 2:
        return None

    x = points_mm[:, 0].astype(np.float64)
    y = points_mm[:, 1].astype(np.float64)

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    if len(x) < 2:
        return None

    unique_x = np.unique(x)
    if len(unique_x) < 2:
        return None

    x_mean = float(np.mean(x))
    x_std = float(np.std(x))

    if not np.isfinite(x_mean) or not np.isfinite(x_std) or x_std < 1e-12:
        return None

    x_normalized = (x - x_mean) / x_std

    slope_normalized = 0.0
    intercept_normalized = float(np.mean(y))
    learning_rate = 0.05
    iterations = 8000
    previous_loss = np.inf

    for _ in range(iterations):
        y_pred = slope_normalized * x_normalized + intercept_normalized
        error = y_pred - y
        loss = float(np.mean(error * error))

        if not np.isfinite(loss):
            return None

        if abs(previous_loss - loss) < 1e-12:
            break

        previous_loss = loss

        gradient_slope = float(2.0 * np.mean(error * x_normalized))
        gradient_intercept = float(2.0 * np.mean(error))

        slope_normalized -= learning_rate * gradient_slope
        intercept_normalized -= learning_rate * gradient_intercept

        if not np.isfinite(slope_normalized) or not np.isfinite(intercept_normalized):
            return None

    slope = slope_normalized / x_std
    intercept = intercept_normalized - slope * x_mean

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None

    return float(slope), float(intercept)



@dataclass(frozen=True)
class Interface2CurveModel:
    """
    Modelo matematico de ajuste usado exclusivamente por la interfaz .

    - theil_sen:   y = mx + b, recta robusta frente a valores atipicos
    - quadratic:   y = ax^2 + bx + c
    - logarithmic: y = a ln(-x) + b, apropiado para el sector III (x < 0)
    """

    method: str
    coefficients: tuple[float, ...]
    equation: str


def _finite_curve_points(points_mm: np.ndarray) -> np.ndarray:
    points = np.asarray(points_mm, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)

    finite_mask = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    return points[finite_mask]


def theil_sen_regression(points_mm: np.ndarray) -> Interface2CurveModel | None:
    """
    Ajuste lineal robusto de Theil-Sen:
        y = mx + b

    La pendiente se obtiene con la mediana de las pendientes entre pares
    de puntos. El intercepto se calcula con la mediana de y - mx.
    """
    points = _finite_curve_points(points_mm)
    if len(points) < 2 or len(np.unique(points[:, 0])) < 2:
        return None

    x = points[:, 0]
    y = points[:, 1]

    slopes: list[float] = []
    for index in range(len(x) - 1):
        dx = x[index + 1:] - x[index]
        dy = y[index + 1:] - y[index]
        valid_mask = np.abs(dx) > 1e-12
        if np.any(valid_mask):
            slopes.extend((dy[valid_mask] / dx[valid_mask]).tolist())

    if not slopes:
        return None

    slope = float(np.median(np.asarray(slopes, dtype=np.float64)))
    intercept = float(np.median(y - slope * x))

    if not np.all(np.isfinite([slope, intercept])):
        return None

    return Interface2CurveModel(
        method="theil_sen",
        coefficients=(slope, intercept),
        equation=f"Y = {slope:.4f}X {' + ' if intercept >= 0 else ' - '}{abs(intercept):.4f}",
    )


def quadratic_regression(points_mm: np.ndarray) -> Interface2CurveModel | None:
    """
    Ajuste parabolico por minimos cuadrados:
        y = ax^2 + bx + c
    """
    points = _finite_curve_points(points_mm)
    if len(points) < 3 or len(np.unique(points[:, 0])) < 3:
        return None

    x = points[:, 0]
    y = points[:, 1]

    try:
        a, b, c = np.polyfit(x, y, 2)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None

    if not np.all(np.isfinite([a, b, c])):
        return None

    return Interface2CurveModel(
        method="quadratic",
        coefficients=(float(a), float(b), float(c)),
        equation=f"Y = {a:.4f}X^2 {' + ' if b >= 0 else ' - '}{abs(b):.4f}X {' + ' if c >= 0 else ' - '}{abs(c):.4f}",
    )


def logarithmic_regression(points_mm: np.ndarray) -> Interface2CurveModel | None:
    """
    Ajuste logaritmico linealizado para el sector III:
        y = a ln(-x) + b

    Se usa -x porque los puntos validos del sector III tienen X negativo.
    """
    points = _finite_curve_points(points_mm)
    if len(points) < 2:
        return None

    x = points[:, 0]
    y = points[:, 1]

    valid_mask = x < -1e-12
    x = x[valid_mask]
    y = y[valid_mask]

    if len(x) < 2:
        return None

    transformed_x = np.log(-x)
    if len(np.unique(np.round(transformed_x, 12))) < 2:
        return None

    design = np.column_stack((transformed_x, np.ones_like(transformed_x)))

    try:
        solution, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a = float(solution[0])
    b = float(solution[1])

    if not np.all(np.isfinite([a, b])):
        return None

    return Interface2CurveModel(
        method="logarithmic",
        coefficients=(a, b),
        equation=f"Y = {a:.4f} ln(-X) {' + ' if b >= 0 else ' - '}{abs(b):.4f}",
    )



def fit_curve_interface_2(
    points_mm: np.ndarray,
    method: str,
) -> Interface2CurveModel | None:
    if method == "theil_sen":
        return theil_sen_regression(points_mm)
    if method == "quadratic":
        return quadratic_regression(points_mm)
    if method == "logarithmic":
        return logarithmic_regression(points_mm)
    raise ValueError(f"Modelo matematico no soportado en interfaz 2: {method}")


def predict_curve_interface_2(
    model: Interface2CurveModel,
    x_values: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float64)

    if model.method == "theil_sen":
        slope, intercept = model.coefficients
        return slope * x + intercept

    if model.method == "quadratic":
        a, b, c = model.coefficients
        return a * x * x + b * x + c

    if model.method == "logarithmic":
        a, b = model.coefficients
        result = np.full_like(x, np.nan, dtype=np.float64)
        valid_mask = x < -1e-12
        result[valid_mask] = a * np.log(-x[valid_mask]) + b
        return result

    raise ValueError(f"Modelo matematico no soportado en interfaz 2: {model.method}")


def sample_curve_interface_2(
    model: Interface2CurveModel,
    points_mm: np.ndarray,
    sample_count: int = 120,
    reverse: bool = False,
) -> np.ndarray:
    """
    Muestrea la curva dentro del intervalo X cubierto por los puntos originales.
    Esto permite dibujarla en OpenCV y convertirla en segmentos G-code.
    """
    points = _finite_curve_points(points_mm)
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float64)

    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))

    if model.method == "logarithmic":
        x_max = min(x_max, -1e-9)

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return np.empty((0, 2), dtype=np.float64)

    if reverse:
        x_values = np.linspace(x_max, x_min, max(2, int(sample_count)))
    else:
        x_values = np.linspace(x_min, x_max, max(2, int(sample_count)))

    y_values = predict_curve_interface_2(model, x_values)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)

    return np.column_stack((x_values[finite_mask], y_values[finite_mask]))


def fit_line(points_mm: np.ndarray, method: str) -> tuple[float, float] | None:
    if method == "polyfit":
        return linear_regression(points_mm)
    if method == "least_squares":
        return least_squares_regression(points_mm)
    if method == "gradient_descent":
        return gradient_descent_regression(points_mm)
    raise ValueError(f"Metodo de regresion no soportado: {method}")


# ============================================================
# DIBUJO Y GRAFICA
# ============================================================

def draw_regression_line(
    frame: np.ndarray,
    mapper: CoordinateMapper,
    slope: float,
    intercept: float,
    points_mm: np.ndarray,
) -> None:
    if len(points_mm) < 2:
        return

    x_min = float(np.min(points_mm[:, 0]))
    x_max = float(np.max(points_mm[:, 0]))

    x_line = np.array([x_min, x_max], dtype=np.float64)
    y_line = slope * x_line + intercept

    line_mm = np.column_stack((x_line, y_line))
    line_px = mapper.machine_to_pixel(line_mm)

    p1 = tuple(np.round(line_px[0]).astype(int))
    p2 = tuple(np.round(line_px[1]).astype(int))

    cv2.line(frame, p1, p2, (0, 255, 255), 2)


def draw_origin_and_roi(
    frame: np.ndarray,
    origin_px: np.ndarray | None,
    roi: tuple[int, int, int, int] | None,
) -> None:
    if roi is not None:
        x1, y1, x2, y2 = normalize_roi(roi, frame.shape)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            frame,
            "ROI camara",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if origin_px is not None:
        ox, oy = np.round(origin_px).astype(int)
        cv2.rectangle(frame, (ox - 10, oy - 10), (ox + 10, oy + 10), (0, 255, 0), 2)
        cv2.drawMarker(frame, (ox, oy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
        cv2.putText(
            frame,
            "0,0",
            (ox + 12, oy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )



def draw_curve_interface_2(
    frame: np.ndarray,
    mapper: CoordinateMapper,
    model: Interface2CurveModel,
    points_mm: np.ndarray,
) -> None:
    """
    Dibuja el modelo matematico de la interfaz  sobre el frame original.
    La vista completa se redimensiona despues dentro del dashboard.
    """
    curve_mm = sample_curve_interface_2(
        model=model,
        points_mm=points_mm,
        sample_count=180,
    )

    if len(curve_mm) < 2:
        return

    curve_px = mapper.machine_to_pixel(curve_mm)
    finite_mask = np.isfinite(curve_px[:, 0]) & np.isfinite(curve_px[:, 1])
    curve_px = curve_px[finite_mask]

    if len(curve_px) < 2:
        return

    polyline = np.round(curve_px).astype(np.int32).reshape((-1, 1, 2))

    cv2.polylines(frame, [polyline], False, (255, 0, 255), 4, cv2.LINE_AA)
    cv2.polylines(frame, [polyline], False, (255, 255, 255), 1, cv2.LINE_AA)


def draw_origin_and_roi_interface_2(
    frame: np.ndarray,
    origin_px: np.ndarray | None,
    roi: tuple[int, int, int, int] | None,
) -> None:
    if roi is not None:
        x1, y1, x2, y2 = normalize_roi(roi, frame.shape)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 3)

        corner_size = 28
        corners = [
            ((x1, y1), (x1 + corner_size, y1), (x1, y1 + corner_size)),
            ((x2, y1), (x2 - corner_size, y1), (x2, y1 + corner_size)),
            ((x1, y2), (x1 + corner_size, y2), (x1, y2 - corner_size)),
            ((x2, y2), (x2 - corner_size, y2), (x2, y2 - corner_size)),
        ]

        for origin, horizontal, vertical in corners:
            cv2.line(frame, origin, horizontal, (0, 255, 255), 4)
            cv2.line(frame, origin, vertical, (0, 255, 255), 4)

        cv2.putText(
            frame,
            "ZONA DE LECTURA",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if origin_px is not None:
        ox, oy = np.round(origin_px).astype(int)
        diamond = np.array(
            [[ox, oy - 18], [ox + 18, oy], [ox, oy + 18], [ox - 18, oy]],
            dtype=np.int32,
        )
        cv2.polylines(frame, [diamond], True, (255, 255, 0), 3)
        cv2.circle(frame, (ox, oy), 5, (255, 255, 255), -1)
        cv2.putText(
            frame,
            "CERO CNC",
            (ox + 22, oy + 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


def draw_interface_2_dashboard(
    frame: np.ndarray,
    origin_px: np.ndarray | None,
    roi: tuple[int, int, int, int] | None,
    points_px: np.ndarray,
    points_px_used: np.ndarray,
    points_mm_used: np.ndarray,
    current_method: str,
    curve_equation: str | None,
    mapper_missing: bool,
    selection_mode: str | None,
    current_camera_index: int | None,
    layout_state: RuntimeSelection | None = None,
) -> np.ndarray:
    """
    Interfaz visual  separada:
    - Lado izquierdo: solo vista de camara.
    - Lado derecho: panel de controles, metodo activo y estado.
    """
    source_height, source_width = frame.shape[:2]

    camera_view = frame.copy()

    draw_origin_and_roi_interface_2(camera_view, origin_px, roi)

    for x, y in points_px:
        cv2.circle(camera_view, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.circle(camera_view, (int(x), int(y)), 9, (255, 255, 255), 1)

    for x, y in points_px_used:
        cv2.circle(camera_view, (int(x), int(y)), 12, (255, 255, 0), 2)

    # ============================================================
    # CANVAS GENERAL: CAMARA IZQUIERDA + PANEL DERECHO
    # ============================================================

    canvas_height = max(720, source_height)
    camera_area_width = max(900, int(round(canvas_height * source_width / max(1, source_height))))
    side_panel_width = 440
    canvas_width = camera_area_width + side_panel_width

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = (12, 14, 22)

    panel_x = camera_area_width

    # Fondos
    cv2.rectangle(canvas, (0, 0), (camera_area_width, canvas_height), (10, 12, 18), -1)
    cv2.rectangle(canvas, (panel_x, 0), (canvas_width, canvas_height), (24, 20, 38), -1)
    cv2.line(canvas, (panel_x, 0), (panel_x, canvas_height), (0, 255, 255), 3)

    # Encabezado de la vista de camara
    header_height = 58
    cv2.rectangle(canvas, (0, 0), (camera_area_width, header_height), (34, 12, 42), -1)
    cv2.putText(
        canvas,
        "VISTA DE CAMARA",
        (22, 38),
        cv2.FONT_HERSHEY_DUPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Ajuste de la imagen de camara dentro del espacio izquierdo
    margin = 18
    available_width = camera_area_width - 2 * margin
    available_height = canvas_height - header_height - 2 * margin

    scale = min(
        available_width / max(1, source_width),
        available_height / max(1, source_height),
    )

    display_width = max(1, int(round(source_width * scale)))
    display_height = max(1, int(round(source_height * scale)))

    resized_camera = cv2.resize(camera_view, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

    camera_x = margin + (available_width - display_width) // 2
    camera_y = header_height + margin + (available_height - display_height) // 2

    canvas[camera_y:camera_y + display_height, camera_x:camera_x + display_width] = resized_camera

    # Marco exclusivo para la zona de camara
    cv2.rectangle(
        canvas,
        (camera_x - 2, camera_y - 2),
        (camera_x + display_width + 2, camera_y + display_height + 2),
        (0, 255, 255),
        2,
    )

    if layout_state is not None:
        layout_state.interface_2_view_box = (
            camera_x,
            camera_y,
            camera_x + display_width,
            camera_y + display_height,
        )
        layout_state.interface_2_source_shape = (source_height, source_width)

    # ============================================================
    # FUNCIONES PARA TEXTO DEL PANEL DERECHO
    # ============================================================

    panel_padding = 24
    text_x = panel_x + panel_padding
    text_right = canvas_width - panel_padding
    text_width = text_right - text_x

    def fit_text_to_width(
        text: str,
        font: int,
        scale_value: float,
        thickness: int,
        max_width: int,
    ) -> str:
        safe_text = str(text)
        text_size, _ = cv2.getTextSize(safe_text, font, scale_value, thickness)
        if text_size[0] <= max_width:
            return safe_text

        ellipsis = "..."
        while len(safe_text) > 3:
            candidate = safe_text[:-1] + ellipsis
            text_size, _ = cv2.getTextSize(candidate, font, scale_value, thickness)
            if text_size[0] <= max_width:
                return candidate
            safe_text = safe_text[:-1]

        return ellipsis

    def put_fit(
        text: str,
        x: int,
        y: int,
        font: int,
        scale_value: float,
        color: tuple[int, int, int],
        thickness: int = 1,
        max_width: int | None = None,
    ) -> None:
        allowed_width = text_width if max_width is None else max_width
        cv2.putText(
            canvas,
            fit_text_to_width(text, font, scale_value, thickness, allowed_width),
            (x, y),
            font,
            scale_value,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_card(y: int, title: str, value: str, accent: tuple[int, int, int]) -> int:
        card_height = 72
        cv2.rectangle(
            canvas,
            (text_x - 10, y - 20),
            (text_right + 10, y + card_height - 18),
            (34, 30, 52),
            -1,
        )
        cv2.rectangle(
            canvas,
            (text_x - 10, y - 20),
            (text_right + 10, y + card_height - 18),
            accent,
            1,
        )

        put_fit(title.upper(), text_x, y, cv2.FONT_HERSHEY_SIMPLEX, 0.44, accent, 1)
        put_fit(value, text_x, y + 32, cv2.FONT_HERSHEY_DUPLEX, 0.66, (255, 255, 255), 2)

        return y + card_height + 8

    # ============================================================
    # PANEL DERECHO
    # ============================================================

    cv2.rectangle(canvas, (panel_x, 0), (canvas_width, 76), (42, 18, 54), -1)

    put_fit(
        "INTERFAZ ",
        text_x,
        32,
        cv2.FONT_HERSHEY_DUPLEX,
        0.78,
        (255, 255, 255),
        2,
    )
    put_fit(
        "Modelos matematicos de ajuste",
        text_x,
        58,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        1,
    )

    y_panel = 110

    camera_text = str(current_camera_index) if current_camera_index is not None else "desconocida"

    y_panel = draw_card(y_panel, "Metodo seleccionado", current_method, (0, 255, 255))
    y_panel = draw_card(y_panel, "Puntos detectados ROI", str(len(points_px)), (255, 255, 0))
    y_panel = draw_card(y_panel, "Puntos usados", str(len(points_mm_used)), (0, 220, 120))
    y_panel = draw_card(y_panel, "Camara activa", camera_text, (255, 160, 80))

    # Estado
    if mapper_missing:
        status = "Origen automatico no disponible"
        status_color = (0, 180, 255)
    elif curve_equation is not None:
        status = curve_equation
        status_color = (0, 255, 180)
    else:
        status = "Minimo 2 puntos validos"
        status_color = (0, 180, 255)

    y_panel += 8
    cv2.line(canvas, (text_x, y_panel), (text_right, y_panel), (105, 105, 135), 1)
    y_panel += 34

    put_fit("ESTADO", text_x, y_panel, cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)
    put_fit(status, text_x, y_panel + 34, cv2.FONT_HERSHEY_SIMPLEX, 0.56, status_color, 2)
    y_panel += 72

    if selection_mode is not None:
        put_fit(
            f"Seleccion activa: {selection_mode}",
            text_x,
            y_panel,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 0),
            2,
        )
        y_panel += 36

    # Metodos disponibles
    y_panel += 8
    cv2.line(canvas, (text_x, y_panel), (text_right, y_panel), (105, 105, 135), 1)
    y_panel += 34
    put_fit("MODELOS DISPONIBLES", text_x, y_panel, cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 0), 2)
    y_panel += 32

    for method_name in SECONDARY_METHODS:
        is_active = method_name == current_method
        color = (0, 255, 180) if is_active else (220, 220, 225)
        marker = ">" if is_active else "-"
        put_fit(
            f"{marker} {method_name}",
            text_x + 4,
            y_panel,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2 if is_active else 1,
        )
        y_panel += 27

    # Controles fijos abajo
    controls = [
        "0,0: automatico abajo izquierda",
        "J: seleccionar ROI",
        "K: guardar trayectoria + GRBL",
        "N: cambiar modelo",
        "V: cambiar camara",
        "X: salir",
    ]

    controls_block_height = 36 + len(controls) * 28
    y_controls = max(y_panel + 24, canvas_height - controls_block_height - 24)

    cv2.line(canvas, (text_x, y_controls - 20), (text_right, y_controls - 20), (105, 105, 135), 1)
    put_fit("CONTROLES", text_x, y_controls, cv2.FONT_HERSHEY_DUPLEX, 0.56, (255, 255, 0), 2)

    y_key = y_controls + 34
    for control in controls:
        put_fit(control, text_x, y_key, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1)
        y_key += 28

    return canvas


def save_regression_plot(
    points_mm: np.ndarray,
    slope: float,
    intercept: float,
    output_path: str,
    method_label: str,
) -> None:
    x = points_mm[:, 0].astype(np.float64)
    y = points_mm[:, 1].astype(np.float64)

    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = slope * x_line + intercept

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="royalblue", label="Puntos usados sector III")
    plt.plot(x_line, y_line, color="crimson", label=f"Regresion lineal ({method_label})")
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    plt.axvline(0, color="black", linewidth=0.8, alpha=0.35)
    plt.xlabel("X maquina (mm)")
    plt.ylabel("Y maquina (mm)")
    plt.title(f"Regresion lineal en coordenadas reales CNC ({method_label})")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()

SECONDARY_METHODS = ("theil_sen", "quadratic", "logarithmic")
# ============================================================
# G-CODE PARA GRBL 1.1f
# ============================================================

def get_unique_finite_points(points_mm: np.ndarray) -> np.ndarray:
    """
    Conserva únicamente coordenadas finitas y elimina duplicados exactos.
    Los puntos permanecen en el mismo sistema absoluto usado por la regresión.
    """
    points = np.asarray(points_mm, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Los puntos deben tener forma Nx2.")

    finite_mask = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    finite_points = points[finite_mask]

    if len(finite_points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    unique_points: list[tuple[float, float]] = []
    used: set[tuple[float, float]] = set()

    for x_point, y_point in finite_points:
        key = (round(float(x_point), 6), round(float(y_point), 6))
        if key in used:
            continue

        used.add(key)
        unique_points.append((float(x_point), float(y_point)))

    return np.asarray(unique_points, dtype=np.float64)


def order_points_from_nearest(
    points_mm: np.ndarray,
    start_xy: tuple[float, float],
) -> np.ndarray:
    """
    Ordena los puntos comenzando por el más cercano al final de la recta.
    Solo reduce recorridos en vacío; no modifica ninguna coordenada.
    """
    if len(points_mm) <= 1:
        return points_mm.copy()

    remaining = [point.copy() for point in points_mm]
    ordered: list[np.ndarray] = []
    current = np.asarray(start_xy, dtype=np.float64)

    while remaining:
        distances = [float(np.linalg.norm(point - current)) for point in remaining]
        nearest_index = int(np.argmin(distances))
        current = remaining.pop(nearest_index)
        ordered.append(current)

    return np.asarray(ordered, dtype=np.float64)


def append_x_marker_gcode(
    gcode_lines: list[str],
    x_center: float,
    y_center: float,
    marker_size: float,
    feed_rate: float,
    plunge_feed_rate: float,
    z_safe: float,
    z_work: float,
    point_index: int,
) -> None:
    """
    Dibuja una X centrada en una coordenada absoluta.

    G90 fija el centro exacto del punto.
    G91 se usa únicamente para los desplazamientos pequeños de cada diagonal.
    Después de cada diagonal se restaura G90 para impedir acumulaciones.
    """
    double_size = 2.0 * marker_size

    gcode_lines.extend(
        [
            f"; Punto original {point_index}: centro X{x_center:.3f} Y{y_center:.3f}",
            "G90",
            f"G0 Z{z_safe:.3f}",
            f"G0 X{x_center:.3f} Y{y_center:.3f}",
            "; Primera diagonal de la X",
            "G91",
            f"G0 X{-marker_size:.3f} Y{-marker_size:.3f}",
            "G90",
            f"G1 Z{z_work:.3f} F{plunge_feed_rate:.1f}",
            "G91",
            f"G1 X{double_size:.3f} Y{double_size:.3f} F{feed_rate:.1f}",
            "G90",
            f"G0 Z{z_safe:.3f}",
            f"G0 X{x_center:.3f} Y{y_center:.3f}",
            "; Segunda diagonal de la X",
            "G91",
            f"G0 X{-marker_size:.3f} Y{marker_size:.3f}",
            "G90",
            f"G1 Z{z_work:.3f} F{plunge_feed_rate:.1f}",
            "G91",
            f"G1 X{double_size:.3f} Y{-double_size:.3f} F{feed_rate:.1f}",
            "G90",
            f"G0 Z{z_safe:.3f}",
        ]
    )


def save_grbl_line_gcode(
    points_mm: np.ndarray,
    slope: float,
    intercept: float,
    output_path: str,
    feed_rate: float,
    plunge_feed_rate: float,
    z_safe: float,
    z_work: float,
    method_label: str,
    return_origin: bool,
    point_marker_size: float = 1.2,
) -> None:
    """
    Genera G-code para dibujar la recta y una X sobre cada punto original usado.

    La trayectoria principal y los centros de los puntos usan coordenadas
    absolutas. Los pequeños trazos de cada X usan coordenadas relativas locales.
    """
    points_mm_valid = get_unique_finite_points(points_mm)

    if len(points_mm_valid) < 2:
        raise ValueError("Se requieren al menos 2 puntos válidos para generar G-code.")

    slope = float(slope)
    intercept = float(intercept)

    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError("La pendiente o el intercepto no son finitos.")

    marker_size = max(0.05, abs(float(point_marker_size)))

    x_values = points_mm_valid[:, 0]
    x_start = float(np.max(x_values))
    x_end = float(np.min(x_values))

    y_start = float(slope * x_start + intercept)
    y_end = float(slope * x_end + intercept)

    if not all(np.isfinite(value) for value in [x_start, y_start, x_end, y_end]):
        raise ValueError("La recta generó coordenadas no finitas.")

    ordered_points = order_points_from_nearest(
        points_mm=points_mm_valid,
        start_xy=(x_end, y_end),
    )

    gcode_lines = [
        f"; Generado por app.py - regresion lineal en coordenadas CNC ({method_label})",
        "; Convencion: sector III = X negativo, Y negativo. Z positivo = arriba.",
        "; Primero se dibuja la recta. Luego se dibuja una X centrada en cada punto original.",
        "G21",       # unidades en mm
        "G90",       # coordenadas absolutas
        "G17",       # plano XY
        "G94",       # avance por minuto
        "G54",       # sistema de coordenadas de trabajo
        f"G0 Z{z_safe:.3f}",
        "; Trazo de la recta de regresion",
        f"G0 X{x_start:.3f} Y{y_start:.3f}",
        f"G1 Z{z_work:.3f} F{plunge_feed_rate:.1f}",
        f"G1 X{x_end:.3f} Y{y_end:.3f} F{feed_rate:.1f}",
        f"G0 Z{z_safe:.3f}",
        "; X centradas sobre todos los puntos originales usados",
    ]

    for index, (x_point, y_point) in enumerate(ordered_points, start=1):
        append_x_marker_gcode(
            gcode_lines=gcode_lines,
            x_center=float(x_point),
            y_center=float(y_point),
            marker_size=marker_size,
            feed_rate=feed_rate,
            plunge_feed_rate=plunge_feed_rate,
            z_safe=z_safe,
            z_work=z_work,
            point_index=index,
        )

    gcode_lines.append("G90")

    if return_origin:
        gcode_lines.append("G0 X0.000 Y0.000")

    gcode_lines.append("M2")

    with open(output_path, "w", encoding="utf-8") as gcode_file:
        gcode_file.write("\n".join(gcode_lines) + "\n")


def save_curve_plot_interface_2(
    points_mm: np.ndarray,
    model: Interface2CurveModel,
    output_path: str,
) -> None:
    """
    Guarda una grafica del modelo matematico seleccionado en la interfaz 2.
    """
    points = _finite_curve_points(points_mm)
    curve_mm = sample_curve_interface_2(
        model=model,
        points_mm=points,
        sample_count=240,
    )

    if len(points) < 2 or len(curve_mm) < 2:
        raise ValueError("No hay suficientes puntos validos para graficar la curva.")

    plt.figure(figsize=(8, 5))
    plt.scatter(points[:, 0], points[:, 1], color="royalblue", label="Puntos originales")
    plt.plot(curve_mm[:, 0], curve_mm[:, 1], color="crimson", label=model.equation)
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    plt.axvline(0, color="black", linewidth=0.8, alpha=0.35)
    plt.xlabel("X maquina (mm)")
    plt.ylabel("Y maquina (mm)")
    plt.title(f"Modelo matematico CNC - {model.method}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def save_grbl_curve_gcode_interface_2(
    points_mm: np.ndarray,
    model: Interface2CurveModel,
    output_path: str,
    feed_rate: float,
    plunge_feed_rate: float,
    z_safe: float,
    z_work: float,
    return_origin: bool,
    point_marker_size: float = 1.2,
    curve_samples: int = 90,
) -> None:
    """
    Genera G-code exclusivo para la interfaz 2.

    Primero aproxima el modelo seleccionado mediante segmentos cortos G1.
    Despues dibuja una X centrada sobre cada punto original utilizado.
    """
    points_mm_valid = get_unique_finite_points(points_mm)
    if len(points_mm_valid) < 2:
        raise ValueError("Se requieren al menos 2 puntos validos para generar G-code.")

    curve_mm = sample_curve_interface_2(
        model=model,
        points_mm=points_mm_valid,
        sample_count=max(12, int(curve_samples)),
        reverse=True,
    )

    if len(curve_mm) < 2:
        raise ValueError("No se pudo muestrear una trayectoria valida para el modelo seleccionado.")

    marker_size = max(0.05, abs(float(point_marker_size)))
    x_start, y_start = curve_mm[0]
    x_end, y_end = curve_mm[-1]

    ordered_points = order_points_from_nearest(
        points_mm=points_mm_valid,
        start_xy=(float(x_end), float(y_end)),
    )

    gcode_lines = [
        f"; Generado por app.py - modelo matematico CNC ({model.method})",
        f"; Ecuacion: {model.equation}",
        "; Primero se dibuja la curva mediante segmentos G1.",
        "; Luego se dibuja una X centrada en cada punto original.",
        "G21",
        "G90",
        "G17",
        "G94",
        "G54",
        f"G0 Z{z_safe:.3f}",
        "; Inicio de la curva",
        f"G0 X{float(x_start):.3f} Y{float(y_start):.3f}",
        f"G1 Z{z_work:.3f} F{plunge_feed_rate:.1f}",
    ]

    for x_curve, y_curve in curve_mm[1:]:
        gcode_lines.append(
            f"G1 X{float(x_curve):.3f} Y{float(y_curve):.3f} F{feed_rate:.1f}"
        )

    gcode_lines.extend(
        [
            f"G0 Z{z_safe:.3f}",
            "; X centradas sobre todos los puntos originales usados",
        ]
    )

    for index, (x_point, y_point) in enumerate(ordered_points, start=1):
        append_x_marker_gcode(
            gcode_lines=gcode_lines,
            x_center=float(x_point),
            y_center=float(y_point),
            marker_size=marker_size,
            feed_rate=feed_rate,
            plunge_feed_rate=plunge_feed_rate,
            z_safe=z_safe,
            z_work=z_work,
            point_index=index,
        )

    gcode_lines.append("G90")

    if return_origin:
        gcode_lines.append("G0 X0.000 Y0.000")

    gcode_lines.append("M2")

    with open(output_path, "w", encoding="utf-8") as gcode_file:
        gcode_file.write("\n".join(gcode_lines) + "\n")

# ============================================================
# SERIAL / GRBL
# ============================================================

def detect_arduino_serial_port() -> str | None:
    """
    Busca un puerto serial compatible con Arduino/GRBL.
    Si no encuentra una descripcion conocida, usa el primer puerto disponible.
    """
    if list_ports is None:
        return None

    candidates = list(list_ports.comports())
    if not candidates:
        return None

    for port_info in candidates:
        device_text = f"{port_info.device} {port_info.description}".lower()
        if (
            "arduino" in device_text
            or "ch340" in device_text
            or "usb serial" in device_text
            or "wch" in device_text
        ):
            return str(port_info.device)

    return str(candidates[0].device)


def read_grbl_response(ser: Any, timeout_s: float) -> str | None:
    """
    Lee una respuesta no vacia de GRBL hasta que venza el timeout.
    """
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue

        line = raw.decode("utf-8", errors="ignore").strip()
        if line:
            return line

    return None


def send_gcode_to_grbl(
    gcode_path: str,
    port: str | None,
    baud_rate: int,
    line_timeout: float = 22.0,
) -> tuple[bool, str]:
    """
    Envia el archivo G-code linea por linea y espera la confirmacion `ok`
    de GRBL antes de continuar con el siguiente comando.
    """
    if serial is None:
        return False, "Falta pyserial. Instala dependencias con: pip install pyserial"

    selected_port = port or detect_arduino_serial_port()
    if not selected_port:
        return False, "No se encontro un puerto serial para Arduino/GRBL. Usa --grbl-port COMx"

    try:
        with open(gcode_path, "r", encoding="utf-8") as gcode_file:
            commands = [
                line.strip()
                for line in gcode_file
                if line.strip() and not line.lstrip().startswith(";")
            ]
    except OSError as error:
        return False, f"No se pudo leer el archivo G-code: {error}"

    if not commands:
        return False, "El archivo G-code esta vacio."

    try:
        with serial.Serial(selected_port, baud_rate, timeout=0.25) as ser:
            time.sleep(2.0)
            ser.reset_input_buffer()
            ser.reset_output_buffer()

            # Despierta GRBL antes de enviar el archivo.
            ser.write(b"\r\n\r\n")
            time.sleep(0.2)
            ser.reset_input_buffer()

            for command in commands:
                ser.write((command + "\n").encode("ascii", errors="ignore"))

                response = read_grbl_response(ser, timeout_s=line_timeout)
                if response is None:
                    return False, f"Timeout esperando respuesta de GRBL tras: {command}"

                while response.lower() != "ok":
                    if response.lower().startswith("error") or response.lower().startswith("alarm"):
                        return False, f"GRBL respondio {response} con comando: {command}"

                    response = read_grbl_response(ser, timeout_s=line_timeout)
                    if response is None:
                        return False, f"No llego 'ok' de GRBL tras: {command}"

    except serial.SerialException as error:
        return False, f"Error serial en {selected_port}: {error}"

    return True, f"G-code enviado correctamente a GRBL por {selected_port}"


# ============================================================
# CAMARA
# ============================================================

def open_camera(camera_source: str) -> cv2.VideoCapture | None:
    source = camera_source.strip()
    candidates: list[tuple[object, int | None]] = []

    if source.isdigit():
        candidates.append((int(source), cv2.CAP_DSHOW))
    else:
        match = re.fullmatch(r"COM(\d+)", source.upper())
        if match:
            com_number = int(match.group(1))
            if com_number > 0:
                candidates.append((com_number - 1, cv2.CAP_DSHOW))
            candidates.append((com_number, cv2.CAP_DSHOW))
        else:
            candidates.append((source, None))

    for candidate, backend in candidates:
        cap = cv2.VideoCapture(candidate, backend) if backend is not None else cv2.VideoCapture(candidate)
        if cap.isOpened():
            print(f"Camara abierta con fuente: {candidate}")
            return cap
        cap.release()

    if re.fullmatch(r"COM(\d+)", source.upper()):
        print(
            "No se pudo abrir la camara desde ese puerto COM. "
            "En OpenCV normalmente debes usar indice de camara: 0, 1, 2..."
        )

    return None


def scan_available_cameras(max_index: int = 6) -> list[int]:
    found_indices: list[int] = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        ok, _ = cap.read()
        cap.release()
        if ok:
            found_indices.append(index)

    return found_indices


def set_mouse_callback(window_name: str, state: RuntimeSelection) -> None:
    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # En la interfaz 2 la imagen de camara se muestra dentro de un canvas
        # con panel lateral. Por eso el clic se convierte desde coordenadas
        # del canvas a coordenadas reales del frame original de la camara.
        if state.visual_interface == 2:
            if state.interface_2_view_box is None or state.interface_2_source_shape is None:
                return

            view_x1, view_y1, view_x2, view_y2 = state.interface_2_view_box
            source_height, source_width = state.interface_2_source_shape

            if not (view_x1 <= x < view_x2 and view_y1 <= y < view_y2):
                return

            view_width = max(1, view_x2 - view_x1)
            view_height = max(1, view_y2 - view_y1)

            x = int(round((x - view_x1) * (source_width - 1) / max(1, view_width - 1)))
            y = int(round((y - view_y1) * (source_height - 1) / max(1, view_height - 1)))

            x = int(np.clip(x, 0, source_width - 1))
            y = int(np.clip(y, 0, source_height - 1))

        if state.mode == "origin":
            state.origin_px = np.array([x, y], dtype=np.float64)
            state.mode = None
            print(f"Origen 0,0 seleccionado en pixel: ({x}, {y})")

        elif state.mode == "roi":
            if state.roi_first_corner is None:
                state.roi_first_corner = (x, y)
                print(f"Primer punto ROI seleccionado: ({x}, {y})")
            else:
                x1, y1 = state.roi_first_corner
                state.roi = (x1, y1, x, y)
                state.roi_first_corner = None
                state.mode = None
                print(f"ROI seleccionado: ({x1}, {y1}, {x}, {y})")

    cv2.setMouseCallback(window_name, on_mouse)




def clear_console() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_interface_1_instructions() -> None:
    print("Controles interfaz :")
    print("- o: seleccionar origen 0,0. Despues haz clic sobre el rectangulo verde.")
    print("- r: seleccionar ROI. Despues haz clic en dos esquinas del cuadrado blanco.")
    print("- g: guardar grafica PNG, generar G-code y enviarlo a GRBL")
    print("- c: cambiar a la siguiente camara disponible")
    print("- m: cambiar metodo de regresion")
    print("- q: salir")
    print("")
    print("Convencion activa por defecto:")
    print("- Derecha en imagen = X negativo")
    print("- Arriba en imagen = Y negativo")
    print("- Z positivo = arriba")
    print("- Regresion solo con sector III: X<0, Y<0")


def print_interface_2_instructions() -> None:
    print("Controles interfaz 2:")
    print("- origen 0,0: automatico en la esquina inferior izquierda del ROI")
    print("- j: seleccionar ROI")
    print("- k: guardar grafica, generar G-code de trayectoria y enviarlo a la CNC")
    print("- v: cambiar camara")
    print("- n: cambiar modelo matematico de la interfaz 2")
    print("- b: volver a la interfaz 1")
    print("- x: salir")
    print("")
    print("Modelos matematicos disponibles en interfaz 2:")
    print("- theil_sen: recta robusta Y = mX + b")
    print("- quadratic: parabola Y = aX^2 + bX + c")
    print("- logarithmic: logaritmico Y = a ln(-X) + b")
    print("")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        default="0",
        help="Fuente de camara. Usa indice (0,1,2,...) o COMx para intento en Windows.",
    )
    parser.add_argument(
        "--scan-cameras",
        action="store_true",
        help="Escanea indices de camara disponibles y termina.",
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=6,
        help="Indice maximo para escaneo cuando usas --scan-cameras.",
    )

    parser.add_argument(
        "--gcode-output",
        default="regression_line.nc",
        help="Archivo de salida G-code compatible con GRBL 1.1f.",
    )

    parser.add_argument(
        "--mm-per-pixel",
        type=float,
        default=0.20,
        help="Escala general de conversion de pixeles a mm.",
    )
    parser.add_argument(
        "--mm-per-pixel-x",
        type=float,
        default=None,
        help="Escala X especifica. Si no se indica, usa --mm-per-pixel.",
    )
    parser.add_argument(
        "--mm-per-pixel-y",
        type=float,
        default=None,
        help="Escala Y especifica. Si no se indica, usa --mm-per-pixel.",
    )

    parser.add_argument(
        "--origin",
        type=parse_point,
        default=None,
        help="Pixel del origen real 0,0. Formato: x,y. Tambien puedes presionar 'o' y dar clic.",
    )
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=None,
        help="Area de enfoque de la camara. Formato: x1,y1,x2,y2. Tambien puedes presionar 'r' y dar 2 clics.",
    )

    parser.add_argument(
        "--x-positive-right",
        action="store_true",
        help="Invierte la convencion X. Por defecto, derecha en imagen = X negativo.",
    )
    parser.add_argument(
        "--y-down-negative",
        action="store_true",
        help="Invierte la convencion Y. Por defecto, arriba en imagen = Y negativo.",
    )
    parser.add_argument(
        "--no-sector-3",
        action="store_true",
        help="No filtra por sector III. Por defecto usa solo puntos con X<0 y Y<0.",
    )
    parser.add_argument(
        "--sector-tolerance-mm",
        type=float,
        default=0.0,
        help="Tolerancia para sector III. Ejemplo 0.5 exige X<-0.5 y Y<-0.5.",
    )

    parser.add_argument(
        "--feed-rate",
        type=float,
        default=300.0,
        help="Avance XY F en mm/min para el movimiento G1.",
    )
    parser.add_argument(
        "--plunge-feed-rate",
        type=float,
        default=120.0,
        help="Avance Z F en mm/min para bajar hasta Z de trabajo.",
    )
    parser.add_argument(
        "--z-safe",
        type=float,
        default=5.0,
        help="Altura segura. En tu sistema Z positivo sube.",
    )
    parser.add_argument(
        "--z-work",
        type=float,
        default=0.0,
        help="Altura de trabajo sobre el plano.",
    )
    parser.add_argument(
        "--return-origin",
        action="store_true",
        help="Al terminar, sube Z y vuelve a X0 Y0.",
    )

    parser.add_argument(
        "--grbl-port",
        default="",
        help="Puerto serial de GRBL. Ejemplo: COM3. Si se omite, se intenta autodeteccion.",
    )
    parser.add_argument(
        "--grbl-baud",
        type=int,
        default=115200,
        help="Baudrate serial para GRBL 1.1f. Normalmente 115200.",
    )
    parser.add_argument(
        "--no-auto-send-grbl",
        action="store_true",
        help="No envia el G-code al Arduino automaticamente al presionar g.",
    )

    parser.add_argument(
        "--method",
        choices=AVAILABLE_METHODS,
        default="polyfit",
        help="Metodo de regresion para ajustar la recta.",
    )

    args = parser.parse_args()

    if args.scan_cameras:
        indices = scan_available_cameras(max_index=max(0, args.max_camera_index))
        if indices:
            print("Camaras detectadas en indices:", ", ".join(str(value) for value in indices))
            print("Tip: Logitech C270 normalmente aparece como 0 o 1.")
        else:
            print("No se detectaron camaras en el rango indicado.")
        return

    available_camera_indices = scan_available_cameras(max_index=max(0, args.max_camera_index))

    cap = open_camera(args.camera)
    if cap is None:
        print("No se pudo abrir la camara.")
        print("Tip: ejecuta --scan-cameras para encontrar el indice correcto.")
        return

    source_upper = args.camera.strip().upper()
    current_camera_index = int(args.camera) if args.camera.strip().isdigit() else None
    com_match = re.fullmatch(r"COM(\d+)", source_upper)
    if current_camera_index is None and com_match:
        com_number = int(com_match.group(1))
        current_camera_index = max(0, com_number - 1)

    if current_camera_index is None and available_camera_indices:
        current_camera_index = available_camera_indices[0]

    mm_per_pixel_x = args.mm_per_pixel_x if args.mm_per_pixel_x is not None else args.mm_per_pixel
    mm_per_pixel_y = args.mm_per_pixel_y if args.mm_per_pixel_y is not None else args.mm_per_pixel

    state = RuntimeSelection(
        origin_px=np.array(args.origin, dtype=np.float64) if args.origin is not None else None,
        roi=args.roi,
    )

    window_name = "Deteccion de puntos + regresion lineal CNC"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 670)
    set_mouse_callback(window_name, state)

    print("Controles:")
    print("- o: seleccionar origen 0,0. Despues haz clic sobre el rectangulo verde.")
    print("- r: seleccionar ROI. Despues haz clic en dos esquinas del cuadrado blanco.")
    print("- g: guardar grafica PNG, generar G-code y enviarlo a GRBL")
    print("- c: cambiar a la siguiente camara disponible")
    print("- m: cambiar metodo de regresion")
    print("- q: salir")
    print("")
    print("Convencion activa por defecto:")
    print("- Derecha en imagen = X negativo")
    print("- Arriba en imagen = Y negativo")
    print("- Z positivo = arriba")
    print("- Regresion solo con sector III: X<0, Y<0")

    current_method = args.method
    current_secondary_method = SECONDARY_METHODS[0]
    visual_interface = 1
    state.visual_interface = visual_interface

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer un frame de la camara.")
            break

        active_origin_px = (
            get_interface_2_auto_origin(state.roi, frame.shape)
            if visual_interface == 2
            else state.origin_px
        )

        mapper = build_mapper(
            origin_px=active_origin_px,
            mm_per_pixel_x=mm_per_pixel_x,
            mm_per_pixel_y=mm_per_pixel_y,
            x_positive_right=args.x_positive_right,
            y_down_negative=args.y_down_negative,
        )

        points_px = detect_red_points(frame, state.roi)

        points_px_used = np.empty((0, 2), dtype=np.int32)
        points_mm_used = np.empty((0, 2), dtype=np.float64)
        slope = intercept = None
        curve_model: Interface2CurveModel | None = None

        if mapper is not None:
            points_px_used, points_mm_used, _ = select_regression_points(
                points_px=points_px,
                mapper=mapper,
                only_sector_3=not args.no_sector_3,
                sector_tolerance_mm=args.sector_tolerance_mm,
            )

            if visual_interface == 1:
                fit = fit_line(points_mm_used, current_method)
                if fit is not None:
                    slope, intercept = fit
                    draw_regression_line(frame, mapper, slope, intercept, points_mm_used)
            else:
                curve_model = fit_curve_interface_2(points_mm_used, current_secondary_method)
                if curve_model is not None:
                    draw_curve_interface_2(frame, mapper, curve_model, points_mm_used)

        if visual_interface == 1:
            draw_origin_and_roi(frame, state.origin_px, state.roi)

            for x, y in points_px:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            for x, y in points_px_used:
                cv2.circle(frame, (int(x), int(y)), 9, (255, 0, 0), 2)

            cv2.putText(
                frame,
                f"Puntos rojos ROI: {len(points_px)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Usados sector III: {len(points_mm_used)}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 255, 180),
                2,
                cv2.LINE_AA,
            )

            camera_text = (
                f"Camara activa: {current_camera_index}"
                if current_camera_index is not None
                else "Camara activa: desconocida"
            )
            cv2.putText(
                frame,
                camera_text,
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Metodo: {current_method}",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 255, 180),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "   ",
                (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 220, 180),
                2,
                cv2.LINE_AA,
            )

            if state.mode is not None:
                cv2.putText(
                    frame,
                    f"Modo seleccion: {state.mode}",
                    (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if mapper is None:
                cv2.putText(
                    frame,
                    "Falta origen: presiona 'o' y clic en el rectangulo verde",
                    (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif slope is not None and intercept is not None:
                cv2.putText(
                    frame,
                    f"Ymm = {slope:.3f}Xmm + {intercept:.3f}",
                    (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    "Se necesitan al menos 2 puntos validos en sector III",
                    (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
        else:
            frame = draw_interface_2_dashboard(
                frame=frame,
                origin_px=active_origin_px,
                roi=state.roi,
                points_px=points_px,
                points_px_used=points_px_used,
                points_mm_used=points_mm_used,
                current_method=current_secondary_method,
                curve_equation=curve_model.equation if curve_model is not None else None,
                mapper_missing=mapper is None,
                selection_mode=state.mode,
                current_camera_index=current_camera_index,
                layout_state=state,
            )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if 65 <= key <= 90:
            key += 32

        if visual_interface == 1:
            if key == ord("i"):
                visual_interface = 2
                state.visual_interface = visual_interface
                state.mode = None
                state.roi_first_corner = None
                clear_console()
                print("Interfaz visual 2 activa.")
                print("Origen 0,0 automatico: esquina inferior izquierda del ROI.")
                print_interface_2_instructions()

            elif key == ord("g"):
                if mapper is None:
                    print("Primero debes seleccionar el origen 0,0 con la tecla 'o'.")
                    continue

                points_px_used, points_mm_used, _ = select_regression_points(
                    points_px=points_px,
                    mapper=mapper,
                    only_sector_3=not args.no_sector_3,
                    sector_tolerance_mm=args.sector_tolerance_mm,
                )

                fit = fit_line(points_mm_used, current_method)
                if fit is not None:
                    slope, intercept = fit

                    save_regression_plot(
                        points_mm=points_mm_used,
                        slope=slope,
                        intercept=intercept,
                        output_path="regression_plot.png",
                        method_label=current_method,
                    )
                    print("Grafica guardada en regression_plot.png")
                    print(f"Regresion CNC: Y = {slope:.6f}X + {intercept:.6f}")
                    print("Puntos usados en mm:")
                    for x_mm, y_mm in points_mm_used:
                        print(f"  X={x_mm:.3f}, Y={y_mm:.3f}")

                    try:
                        script_dir = os.path.dirname(os.path.abspath(__file__))

                        gcode_output_path = args.gcode_output
                        if not os.path.isabs(gcode_output_path):
                            gcode_output_path = os.path.join(script_dir, gcode_output_path)

                        save_grbl_line_gcode(
                            points_mm=points_mm_used,
                            slope=slope,
                            intercept=intercept,
                            output_path=gcode_output_path,
                            feed_rate=max(1.0, args.feed_rate),
                            plunge_feed_rate=max(1.0, args.plunge_feed_rate),
                            z_safe=args.z_safe,
                            z_work=args.z_work,
                            method_label=current_method,
                            return_origin=args.return_origin,
                        )

                        print(
                            f"G-code GRBL guardado en {gcode_output_path} "
                            "con recta y puntos originales"
                        )

                        if not args.no_auto_send_grbl:
                            ok, message = send_gcode_to_grbl(
                                gcode_path=gcode_output_path,
                                port=args.grbl_port.strip() or None,
                                baud_rate=max(1200, args.grbl_baud),
                            )
                            print(message)
                        else:
                            print("Envio automatico desactivado por --no-auto-send-grbl")

                    except ValueError as error:
                        print(f"No se pudo generar G-code: {error}")
                else:
                    print("No se pudo calcular una regresion estable con los puntos actuales.")
                    print("Verifica que haya al menos 2 puntos rojos dentro del ROI y en sector III.")

            elif key == ord("o"):
                state.mode = "origin"
                state.roi_first_corner = None
                print("Haz clic sobre el rectangulo verde que representa X0 Y0.")

            elif key == ord("r"):
                state.mode = "roi"
                state.roi_first_corner = None
                print("Haz clic en una esquina del cuadrado blanco y luego en la esquina opuesta.")

            elif key == ord("m"):
                current_index = AVAILABLE_METHODS.index(current_method)
                next_index = (current_index + 1) % len(AVAILABLE_METHODS)
                current_method = AVAILABLE_METHODS[next_index]
                print(f"Metodo de regresion activo: {current_method}")

            elif key == ord("c"):
                if not available_camera_indices:
                    available_camera_indices = scan_available_cameras(max_index=max(0, args.max_camera_index))

                if not available_camera_indices:
                    print("No hay camaras disponibles para cambiar.")
                    continue

                if current_camera_index in available_camera_indices:
                    current_position = available_camera_indices.index(current_camera_index)
                    next_position = (current_position + 1) % len(available_camera_indices)
                else:
                    next_position = 0

                next_index = available_camera_indices[next_position]
                new_cap = cv2.VideoCapture(next_index, cv2.CAP_DSHOW)
                opened, _ = new_cap.read()
                if opened:
                    cap.release()
                    cap = new_cap
                    current_camera_index = next_index
                    cv2.setMouseCallback(window_name, lambda *args_lambda: None)
                    set_mouse_callback(window_name, state)
                    print(f"Cambio de camara exitoso. Camara activa: {next_index}")
                else:
                    new_cap.release()
                    print(f"No se pudo cambiar a la camara {next_index}.")

            elif key == ord("q"):
                break

        else:
            if key == ord("k"):
                if mapper is None:
                    print("No se pudo calcular el origen automatico de la interfaz 2.")
                    continue

                points_px_used, points_mm_used, _ = select_regression_points(
                    points_px=points_px,
                    mapper=mapper,
                    only_sector_3=not args.no_sector_3,
                    sector_tolerance_mm=args.sector_tolerance_mm,
                )

                curve_model = fit_curve_interface_2(points_mm_used, current_secondary_method)
                if curve_model is not None:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    plot_output_path = os.path.join(script_dir, "regression_plot_interface_2.png")
                    gcode_output_path = args.gcode_output
                    if not os.path.isabs(gcode_output_path):
                        gcode_output_path = os.path.join(script_dir, gcode_output_path)

                    try:
                        save_curve_plot_interface_2(
                            points_mm=points_mm_used,
                            model=curve_model,
                            output_path=plot_output_path,
                        )
                        print(f"Grafica guardada en {plot_output_path}")
                        print(f"Modelo CNC interfaz  ({curve_model.method}): {curve_model.equation}")
                        print("Puntos usados en mm:")
                        for x_mm, y_mm in points_mm_used:
                            print(f"  X={x_mm:.3f}, Y={y_mm:.3f}")

                        save_grbl_curve_gcode_interface_2(
                            points_mm=points_mm_used,
                            model=curve_model,
                            output_path=gcode_output_path,
                            feed_rate=max(1.0, args.feed_rate),
                            plunge_feed_rate=max(1.0, args.plunge_feed_rate),
                            z_safe=args.z_safe,
                            z_work=args.z_work,
                            return_origin=args.return_origin,
                        )
                        print(
                            f"G-code GRBL guardado en {gcode_output_path} "
                            f"con trayectoria {curve_model.method} y puntos originales"
                        )

                        if not args.no_auto_send_grbl:
                            ok, message = send_gcode_to_grbl(
                                gcode_path=gcode_output_path,
                                port=args.grbl_port.strip() or None,
                                baud_rate=max(1200, args.grbl_baud),
                            )
                            print(message)
                        else:
                            print("Envio automatico desactivado por --no-auto-send-grbl")

                    except ValueError as error:
                        print(f"No se pudo generar la trayectoria o el G-code: {error}")
                else:
                    print(
                        "No se pudo calcular el modelo matematico seleccionado "
                        f"({current_secondary_method})."
                    )
                    print(
                        "Verifica los puntos del ROI. La parabola requiere minimo 3 puntos; "
                        "los modelos logaritmico y exponencial requieren minimo 2."
                    )

            elif key == ord("j"):
                state.mode = "roi"
                state.roi_first_corner = None
                print("Interfaz : haz clic en dos esquinas para definir el ROI.")

            elif key == ord("n"):
                current_index = SECONDARY_METHODS.index(current_secondary_method)
                next_index = (current_index + 1) % len(SECONDARY_METHODS)
                current_secondary_method = SECONDARY_METHODS[next_index]
                print(f"Metodo interfaz  activo: {current_secondary_method}")

            elif key == ord("v"):
                if not available_camera_indices:
                    available_camera_indices = scan_available_cameras(max_index=max(0, args.max_camera_index))

                if not available_camera_indices:
                    print("No hay camaras disponibles para cambiar.")
                    continue

                if current_camera_index in available_camera_indices:
                    current_position = available_camera_indices.index(current_camera_index)
                    next_position = (current_position + 1) % len(available_camera_indices)
                else:
                    next_position = 0

                next_index = available_camera_indices[next_position]
                new_cap = cv2.VideoCapture(next_index, cv2.CAP_DSHOW)
                opened, _ = new_cap.read()
                if opened:
                    cap.release()
                    cap = new_cap
                    current_camera_index = next_index
                    cv2.setMouseCallback(window_name, lambda *args_lambda: None)
                    set_mouse_callback(window_name, state)
                    print(f"Interfaz : cambio de camara exitoso. Camara activa: {next_index}")
                else:
                    new_cap.release()
                    print(f"No se pudo cambiar a la camara {next_index}.")

            elif key == ord("b"):
                visual_interface = 1
                state.visual_interface = visual_interface
                state.interface_2_view_box = None
                state.interface_2_source_shape = None
                state.mode = None
                state.roi_first_corner = None
                clear_console()
                print("Volviste a la interfaz visual .")
                print_interface_1_instructions()

            elif key == ord("x"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()