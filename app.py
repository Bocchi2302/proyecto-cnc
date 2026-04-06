import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import time
from typing import Any

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


def detect_red_points(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
        points.append((x, y))

    if not points:
        return np.empty((0, 2), dtype=np.int32)

    points_array = np.array(points, dtype=np.int32)
    points_array = points_array[np.argsort(points_array[:, 0])]
    return points_array


def linear_regression(points: np.ndarray) -> tuple[float, float] | None:
    if len(points) < 2:
        return None

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)

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


def draw_regression_line(frame: np.ndarray, slope: float, intercept: float) -> None:
    height, width = frame.shape[:2]

    x1, x2 = 0, width - 1
    y1 = int(slope * x1 + intercept)
    y2 = int(slope * x2 + intercept)

    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


def save_regression_plot(points: np.ndarray, slope: float, intercept: float, output_path: str) -> None:
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)

    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = slope * x_line + intercept

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="royalblue", label="Puntos detectados")
    plt.plot(x_line, y_line, color="crimson", label="Regresión lineal")
    plt.xlabel("X (pixeles)")
    plt.ylabel("Y (pixeles)")
    plt.title("Regresión lineal a partir de puntos detectados")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def save_grbl_line_gcode(
    points: np.ndarray,
    slope: float,
    intercept: float,
    output_path: str,
    mm_per_pixel: float,
    feed_rate: float,
) -> None:
    x = points[:, 0].astype(np.float64)
    if len(x) < 2:
        raise ValueError("Se requieren al menos 2 puntos para generar G-code.")

    x_min_px = float(np.min(x))
    x_max_px = float(np.max(x))

    y_min_px = float(slope * x_min_px + intercept)
    y_max_px = float(slope * x_max_px + intercept)

    # Traslada al origen y escala a milimetros para GRBL.
    x1_mm = 0.0
    y1_mm = 0.0
    x2_mm = (x_max_px - x_min_px) * mm_per_pixel
    y2_mm = (y_max_px - y_min_px) * mm_per_pixel

    gcode_lines = [
        "; Generado por app.py - recta por regresion lineal",
        "G21",  # mm
        "G90",  # coordenadas absolutas
        "G17",  # plano XY
        "G94",  # avance por minuto
        "G0 X0 Y0",
        f"G1 X{x2_mm:.3f} Y{y2_mm:.3f} F{feed_rate:.1f}",
        "G0 X0 Y0",
        "M2",
    ]

    with open(output_path, "w", encoding="utf-8") as gcode_file:
        gcode_file.write("\n".join(gcode_lines) + "\n")


def detect_arduino_serial_port() -> str | None:
    if list_ports is None:
        return None

    candidates = list(list_ports.comports())
    if not candidates:
        return None

    # Prioriza dispositivos que suelen corresponder a Arduino/USB-serial.
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


def send_gcode_to_grbl(gcode_path: str, port: str | None, baud_rate: int, line_timeout: float = 3.0) -> tuple[bool, str]:
    if serial is None:
        return False, "Falta pyserial. Instala dependencias con pip install -r requirements.txt"

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
            # Reinicia buffer y despierta GRBL.
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
                if response.lower().startswith("error"):
                    return False, f"GRBL respondio {response} con comando: {command}"
                if response.lower() != "ok":
                    # Algunas placas envian mensajes extra; seguimos esperando ok.
                    while response is not None and response.lower() != "ok":
                        if response.lower().startswith("error"):
                            return False, f"GRBL respondio {response} con comando: {command}"
                        response = read_grbl_response(ser, timeout_s=line_timeout)
                    if response is None:
                        return False, f"No llego 'ok' de GRBL tras: {command}"

    except serial.SerialException as error:
        return False, f"Error serial en {selected_port}: {error}"

    return True, f"G-code enviado correctamente a GRBL por {selected_port}"


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
            "En OpenCV normalmente debes usar indice de camara (0, 1, 2...)."
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
        help="Archivo de salida G-code compatible con GRBL.",
    )
    parser.add_argument(
        "--mm-per-pixel",
        type=float,
        default=0.20,
        help="Escala de conversion de pixeles a mm para exportar G-code.",
    )
    parser.add_argument(
        "--feed-rate",
        type=float,
        default=300.0,
        help="Avance (F) en mm/min para el movimiento G1.",
    )
    parser.add_argument(
        "--grbl-port",
        default="",
        help="Puerto serial de GRBL (ejemplo: COM3). Si se omite, se intenta autodeteccion.",
    )
    parser.add_argument(
        "--grbl-baud",
        type=int,
        default=115200,
        help="Baudrate serial para GRBL (normalmente 115200).",
    )
    parser.add_argument(
        "--no-auto-send-grbl",
        action="store_true",
        help="No envia el G-code al Arduino automaticamente al presionar g.",
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
        print("No se pudo abrir la cámara.")
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

    print("Controles:")
    print("- g: guardar grafica (PNG), generar G-code y enviarlo a GRBL")
    print("- c: cambiar a la siguiente camara disponible")
    print("- q: salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer un frame de la cámara.")
            break

        points = detect_red_points(frame)

        slope = intercept = None
        fit = linear_regression(points)
        if fit is not None:
            slope, intercept = fit
            draw_regression_line(frame, slope, intercept)

        for x, y in points:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        cv2.putText(
            frame,
            f"Puntos detectados: {len(points)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
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

        if slope is not None and intercept is not None:
            cv2.putText(
                frame,
                f"y = {slope:.3f}x + {intercept:.3f}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "Se necesitan al menos 2 puntos",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Deteccion de puntos + regresion lineal", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("g"):
            fit = linear_regression(points)
            if fit is not None:
                slope, intercept = fit
                save_regression_plot(points, slope, intercept, "regression_plot.png")
                print("Grafica guardada en regression_plot.png")
                try:
                    save_grbl_line_gcode(
                        points=points,
                        slope=slope,
                        intercept=intercept,
                        output_path=args.gcode_output,
                        mm_per_pixel=max(1e-9, args.mm_per_pixel),
                        feed_rate=max(1.0, args.feed_rate),
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
