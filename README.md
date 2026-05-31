# Interfaz 2 (modelos matematicos CNC)

Este README documenta la de `app.py`.

La Interfaz  usa la camara para detectar puntos rojos dentro del ROI, convierte a coordenadas CNC y ajusta una trayectoria con uno de estos modelos:

- `theil_sen`: recta robusta, `Y = mX + b`
- `quadratic`: parabola, `Y = aX^2 + bX + c`
- `logarithmic`: curva logaritmica, `Y = a ln(-X) + b`

## Requisitos

- Python 3.10+
- Camara web
- Arduino con GRBL (opcional, para envio automatico)

Instalacion:

```bash
pip install -r requirements.txt
```

## Ejecutar

```bash
python app.py
```

Opcional: camara especifica o escaneo de camaras.

```bash
python app.py --camera 1
python app.py --camera COM3
python app.py --scan-cameras
python app.py --scan-cameras --max-camera-index 10
```

## Activar Interfaz 



Al entrar en Interfaz :

- El origen `0,0` se calcula automaticamente en la esquina inferior izquierda del ROI.
- Si no hay ROI definido todavia, el origen se calcula en la esquina inferior izquierda del frame completo.

## Controles de Interfaz 2

```text
j = seleccionar ROI (dos clics)
k = guardar grafica, generar G-code y enviar a GRBL
n = cambiar modelo matematico
v = cambiar camara
x = salir
```



## Modelos disponibles en Interfaz 

### 1) `theil_sen`

- Ajuste lineal robusto basado en medianas.
- Buena opcion cuando hay ruido moderado.

### 2) `quadratic`

- Ajuste parabolico por minimos cuadrados.
- Requiere al menos 3 puntos validos.

### 3) `logarithmic`

- Ajuste `Y = a ln(-X) + b`.
- Requiere valores de `X < 0` en los puntos usados.
- Requiere al menos 2 puntos validos.

## Sector III y filtrado de puntos

Por defecto, los puntos usados para ajuste se filtran al sector III:

- `X < 0`
- `Y < 0`

Opciones relacionadas:

- `--no-sector-3`: desactiva el filtrado de sector III
- `--sector-tolerance-mm`: tolerancia para exigir, por ejemplo, `X < -0.5` y `Y < -0.5`

## Salida al presionar `k`

Interfaz  genera:

- Grafica: `regression_plot_interface_2.png`
- G-code: archivo definido por `--gcode-output` (por defecto `regression_line.nc`)

El G-code es compatible con GRBL 1.1f.

## Parametros utiles para Interfaz 

```bash
python app.py \
  --camera 0 \
  --roi 540,180,1060,700 \
  --mm-per-pixel 0.20 \
  --feed-rate 300 \
  --plunge-feed-rate 120 \
  --z-safe 5 \
  --z-work 0 \
  --gcode-output regression_line.nc \
  --grbl-port COM3 \
  --grbl-baud 115200
```

Opciones frecuentes:

- `--roi`: ROI inicial
- `--mm-per-pixel`, `--mm-per-pixel-x`, `--mm-per-pixel-y`: escala pixel a mm
- `--feed-rate`: avance XY
- `--plunge-feed-rate`: avance Z
- `--z-safe`, `--z-work`: alturas Z
- `--gcode-output`: nombre/ruta del archivo `.nc`
- `--grbl-port`, `--grbl-baud`: configuracion serial GRBL
- `--no-auto-send-grbl`: no enviar automaticamente al presionar `k`

## Notas de uso

- Si no detecta puntos correctamente, mejora iluminacion y evita sombras fuertes.
- Verifica que los puntos rojos esten dentro del ROI.
- El modelo activo se cambia con `n` y se refleja en el panel derecho de la Interfaz .
