# Detección de puntos con cámara + regresión lineal

Este proyecto usa una cámara web para detectar puntos rojos en tiempo real (sobre una hoja blanca) y ajustar una recta por regresión lineal con esos puntos.

Metodos disponibles de ajuste:

- `polyfit`: ajuste lineal usando `numpy.polyfit`
- `least_squares`: ajuste lineal por minimos cuadrados con `numpy.linalg.lstsq`

## Requisitos

- Python 3.10+
- Cámara web

Instala dependencias:

```bash
pip install -r requirements.txt
```

## Ejecutar

```bash
python app.py
```

Para elegir el metodo de regresion al iniciar:

```bash
python app.py --method least_squares
```

Para indicar una cámara específica:

```bash
python app.py --camera 1
python app.py --camera COM3
```

Si usas `COMx`, el script intenta mapearlo a índice de cámara en Windows.

Para encontrar el índice correcto (recomendado para Logitech C270):

```bash
python app.py --scan-cameras
python app.py --scan-cameras --max-camera-index 10
```

## Controles

- `g`: genera y guarda:
	- gráfica en `regression_plot.png`
	- G-code compatible con GRBL en `regression_line.nc`
	- envio automatico al Arduino con GRBL (serial)
- `c`: cambia a la siguiente cámara disponible
- `m`: cambia entre metodos de regresion disponibles
- `q`: cierra la aplicación

## Exportar para CNC (GRBL en Arduino)

Al presionar `g`, se genera un archivo `.nc` con comandos `G21/G90/G1` compatibles con GRBL.
Despues de generarlo, el script lo envia automaticamente al Arduino por puerto serial.

Puedes ajustar salida, escala y avance:

```bash
python app.py --gcode-output mi_linea.nc --mm-per-pixel 0.25 --feed-rate 350 --grbl-port COM3 --grbl-baud 115200
```

- `--gcode-output`: nombre del archivo G-code de salida
- `--mm-per-pixel`: escala de conversion de pixeles a milimetros
- `--feed-rate`: valor `F` del movimiento lineal (`G1`) en mm/min
- `--grbl-port`: puerto serial de GRBL (si se omite, se intenta autodeteccion)
- `--grbl-baud`: baudrate de GRBL (normalmente `115200`)
- `--no-auto-send-grbl`: desactiva el envio automatico al presionar `g`

Si el firmware reporta error de bloqueo (ALARM), desbloquea GRBL primero desde tu sender con `$X` y vuelve a intentar.

## Nota de uso

La detección está orientada a una hoja blanca con puntos rojos marcados.
Si no detecta bien, mejora la iluminación general y evita sombras fuertes.
