import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
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


AVAILABLE_METHODS = ("polyfit", "least_squares")


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


def fit_line(points_mm: np.ndarray, method: str) -> tuple[float, float] | None:
    if method == "polyfit":
        return linear_regression(points_mm)
    if method == "least_squares":
        return least_squares_regression(points_mm)
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


# ============================================================
# G-CODE PARA GRBL 1.1f
# ============================================================

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
) -> None:
    if len(points_mm) < 2:
        raise ValueError("Se requieren al menos 2 puntos validos para generar G-code.")

    x_values = points_mm[:, 0].astype(np.float64)

    # En sector III, max(X) suele ser el punto mas cercano al origen
    # y min(X) el mas alejado hacia X negativo.
    x_start = float(np.max(x_values))
    x_end = float(np.min(x_values))

    y_start = float(slope * x_start + intercept)
    y_end = float(slope * x_end + intercept)

    if not all(np.isfinite(value) for value in [x_start, y_start, x_end, y_end]):
        raise ValueError("La recta genero coordenadas no finitas.")

    gcode_lines = [
        f"; Generado por app.py - regresion lineal en coordenadas CNC ({method_label})",
        "; Convencion: sector III = X negativo, Y negativo. Z positivo = arriba.",
        "G21",       # unidades en mm
        "G90",       # coordenadas absolutas
        "G17",       # plano XY
        "G94",       # avance por minuto
        "G54",       # sistema de coordenadas de trabajo
        f"G0 Z{z_safe:.3f}",
        f"G0 X{x_start:.3f} Y{y_start:.3f}",
        f"G1 Z{z_work:.3f} F{plunge_feed_rate:.1f}",
        f"G1 X{x_end:.3f} Y{y_end:.3f} F{feed_rate:.1f}",
        f"G0 Z{z_safe:.3f}",
    ]

    if return_origin:
        gcode_lines.append("G0 X0.000 Y0.000")

    gcode_lines.append("M2")

    with open(output_path, "w", encoding="utf-8") as gcode_file:
        gcode_file.write("\n".join(gcode_lines) + "\n")


# ============================================================
# SERIAL / GRBL
# ============================================================

def detect_arduino_serial_port() -> str | None:
    if list_ports is None:
        return None

    candidates = list(list_ports.comports())
    if not candidates:
        return None

    for port_info in candidates:
        device_text = f"{port_info.device} {port_info.description}".lower()
        if "arduino" in device_text or "ch340" in device_text or "usb serial" in device_text:
            return str(port_info.device)

    return str(candidates[0].device)


def read_grbl_response(ser: Any, timeout_s: float) -> str | None:
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
    line_timeout: float = 3.0,
) -> tuple[bool, str]:
    if serial is None:
        return False, "Falta pyserial. Instala dependencias con: pip install pyserial"

    selected_port = port or detect_arduino_serial_port()
    if not selected_port:
        return False, "No se encontro un puerto serial para Arduino/GRBL. Usa --grbl-port COMx"

    try:
        with open(gcode_path, "r", encoding="utf-8") as gcode_file:
            lines = [line.strip() for line in gcode_file if line.strip() and not line.lstrip().startswith(";")]
    except OSError as error:
        return False, f"No se pudo leer el archivo G-code: {error}"

    if not lines:
        return False, "El archivo G-code esta vacio."

    try:
        with serial.Serial(selected_port, baud_rate, timeout=0.25) as ser:
            time.sleep(2.0)
            ser.reset_input_buffer()
            ser.reset_output_buffer()

            ser.write(b"\r\n\r\n")
            time.sleep(0.2)
            ser.reset_input_buffer()

            for command in lines:
                ser.write((command + "\n").encode("ascii", errors="ignore"))

                response = read_grbl_response(ser, timeout_s=line_timeout)
                if response is None:
                    return False, f"Timeout esperando respuesta de GRBL tras: {command}"

                if response.lower().startswith("error") or response.lower().startswith("alarm"):
                    return False, f"GRBL respondio {response} con comando: {command}"

                if response.lower() != "ok":
                    while response is not None and response.lower() != "ok":
                        if response.lower().startswith("error") or response.lower().startswith("alarm"):
                            return False, f"GRBL respondio {response} con comando: {command}"
                        response = read_grbl_response(ser, timeout_s=line_timeout)

                    if response is None:
                        return False, f"No llego 'ok' de GRBL tras: {command}"

    except serial.SerialException as error:
        return False, f"Error serial en {selected_port}: {error}"

    return True, f"G-code enviado correctamente a GRBL 1.1f por {selected_port}"


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
    cv2.namedWindow(window_name)
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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer un frame de la camara.")
            break

        mapper = build_mapper(
            origin_px=state.origin_px,
            mm_per_pixel_x=mm_per_pixel_x,
            mm_per_pixel_y=mm_per_pixel_y,
            x_positive_right=args.x_positive_right,
            y_down_negative=args.y_down_negative,
        )

        points_px = detect_red_points(frame, state.roi)

        points_px_used = np.empty((0, 2), dtype=np.int32)
        points_mm_used = np.empty((0, 2), dtype=np.float64)
        slope = intercept = None

        if mapper is not None:
            points_px_used, points_mm_used, _ = select_regression_points(
                points_px=points_px,
                mapper=mapper,
                only_sector_3=not args.no_sector_3,
                sector_tolerance_mm=args.sector_tolerance_mm,
            )

            fit = fit_line(points_mm_used, current_method)
            if fit is not None:
                slope, intercept = fit
                draw_regression_line(frame, mapper, slope, intercept, points_mm_used)

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

        if state.mode is not None:
            cv2.putText(
                frame,
                f"Modo seleccion: {state.mode}",
                (10, 145),
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

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("g"):
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
                    save_grbl_line_gcode(
                        points_mm=points_mm_used,
                        slope=slope,
                        intercept=intercept,
                        output_path=args.gcode_output,
                        feed_rate=max(1.0, args.feed_rate),
                        plunge_feed_rate=max(1.0, args.plunge_feed_rate),
                        z_safe=args.z_safe,
                        z_work=args.z_work,
                        method_label=current_method,
                        return_origin=args.return_origin,
                    )
                    print(f"G-code GRBL guardado en {args.gcode_output}")

                    if not args.no_auto_send_grbl:
                        ok, message = send_gcode_to_grbl(
                            gcode_path=args.gcode_output,
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
