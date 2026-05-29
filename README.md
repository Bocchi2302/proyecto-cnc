# Detección de puntos con cámara + regresión numérica robusta

Este proyecto usa una cámara web para detectar puntos rojos en tiempo real sobre una hoja blanca, convertirlos a coordenadas reales de una CNC y ajustar una recta usando métodos numéricos robustos disponibles en la interfaz visual .

La interfaz  está pensada para trabajar con métodos de regresión más resistentes al ruido, puntos mal detectados o errores de medición de la cámara.

## Métodos disponibles en la interfaz 2

* `theil_sen`: método robusto que calcula pendientes entre pares de puntos y usa la mediana para obtener una recta estable.
* `ransac`: método robusto que busca la mejor recta descartando puntos atípicos o falsos positivos.
* `orthogonal`: regresión ortogonal que considera el error tanto en X como en Y, útil cuando la cámara puede medir con desviaciones horizontales y verticales.

## Requisitos

* Python 3.10+
* Cámara web
* Arduino con GRBL, si se desea enviar el G-code automáticamente
* Dependencias del proyecto instaladas

Instala dependencias:

```bash
pip install -r requirements.txt
```

## Ejecutar

```bash
python app.py
```

Para indicar una cámara específica:

```bash
python app.py --camera 1
python app.py --camera COM3
```

Si usas `COMx`, el script intenta mapearlo a un índice de cámara en Windows.

Para encontrar el índice correcto de la cámara:

```bash
python app.py --scan-cameras
python app.py --scan-cameras --max-camera-index 10
```

## Controles de la interfaz 

```text
u = seleccionar origen 0,0
j = seleccionar ROI o zona de lectura
k = guardar gráfica, generar G-code y enviarlo a GRBL
n = cambiar método de regresión de la interfaz 
v = cambiar a la siguiente cámara disponible
x = cerrar la aplicación
```

## Uso del origen y del ROI

Antes de generar la regresión, se debe definir el origen real de la CNC y la zona donde se detectan los puntos rojos.

Para seleccionar el origen:

```text
u
```

Luego haz clic sobre el punto que representa el origen `X0 Y0`.

Para seleccionar el ROI:

```text
j
```

Luego haz clic en dos esquinas opuestas de la zona donde están los puntos rojos.

## Cambiar método de regresión

Dentro de la interfaz 2, presiona:

```text
n
```

Cada vez que se presiona `n`, el programa cambia entre los métodos disponibles:

```text
theil_sen
ransac
orthogonal
```

## Comparación rápida de los métodos

| Método       | Cuándo conviene usarlo                                                    |
| ------------ | ------------------------------------------------------------------------- |
| `theil_sen`  | Cuando los puntos están casi alineados, pero tienen ruido moderado.       |
| `ransac`     | Cuando hay puntos falsos, mal detectados o muy alejados de la línea real. |
| `orthogonal` | Cuando el error de medición puede estar tanto en X como en Y.             |

Para pruebas reales con cámara, `ransac` suele ser una buena opción cuando la detección puede generar puntos incorrectos.

## Exportar para CNC con GRBL

Al presionar:

```text
k
```

El programa genera:

* una gráfica PNG con la regresión calculada
* un archivo G-code compatible con GRBL
* el envío automático al Arduino, si está habilitado

El archivo G-code usa comandos compatibles con GRBL como:

```text
G21
G90
G17
G94
G54
G0
G1
M2
```

Puedes ajustar salida, escala, avance y puerto serial:

```bash
python app.py --gcode-output mi_linea.nc --mm-per-pixel 0.25 --feed-rate 350 --grbl-port COM3 --grbl-baud 115200
```

Opciones principales:

* `--gcode-output`: nombre del archivo G-code de salida.
* `--mm-per-pixel`: escala general de conversión de píxeles a milímetros.
* `--mm-per-pixel-x`: escala específica para el eje X.
* `--mm-per-pixel-y`: escala específica para el eje Y.
* `--feed-rate`: velocidad de avance XY en mm/min.
* `--plunge-feed-rate`: velocidad de bajada en Z.
* `--z-safe`: altura segura del eje Z.
* `--z-work`: altura de trabajo.
* `--grbl-port`: puerto serial del Arduino con GRBL.
* `--grbl-baud`: baudrate de GRBL, normalmente `115200`.
* `--no-auto-send-grbl`: desactiva el envío automático al presionar `k`.

Si GRBL reporta un error tipo `ALARM`, desbloquea primero desde tu sender con:

```text
$X
```

Luego vuelve a intentar el envío.

## Nota de uso

La detección está orientada a una hoja blanca con puntos rojos marcados.

Si no detecta bien los puntos:

* mejora la iluminación general
* evita sombras fuertes
* verifica que los puntos estén dentro del ROI
* asegúrate de que el origen `X0 Y0` esté bien seleccionado
* revisa que haya al menos dos puntos válidos en el sector de trabajo

La interfaz  trabaja con métodos robustos, por lo que puede funcionar mejor cuando hay ruido visual o puntos detectados incorrectamente.
