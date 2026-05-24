# Guía de Calibración de Motores CNC

## Parámetros de Z (Herramienta)

Tu código ahora soporta movimientos en Z para controlar la altura de la herramienta:

### Parámetros disponibles:
```bash
--z-up X.X      # Altura Z (en mm) para **levantar** la herramienta durante desplazamientos
--z-down Y.Y    # Altura Z (en mm) para **escribir** (bajar la herramienta)
```

### Valores recomendados para calibración:

| Parámetro | Valor sugerido | Notas |
|-----------|----------------|-------|
| `--z-up` | 5.0 a 10.0 | Altura segura para movimiento sin tocar |
| `--z-down` | 0.0 a -5.0 | Altura de contacto (0=superficie, negativo=presión) |

## Flujo de movimiento generado

El G-code ahora hace lo siguiente:

```gcode
G0 X0 Y0 Z5.0       → Posición inicial con herramienta LEVANTADA
G1 X51.8 Y-5.68 Z0.0 → ESCRIBIR la línea bajando la herramienta  
G0 X0 Y0 Z5.0       → Volver al origen con herramienta LEVANTADA
M2                   → Fin
```

## Cómo calibrar paso a paso

### 1️⃣ **Calibración de Z-UP (levantar herramienta)**
```bash
python app.py --no-auto-send-grbl --z-up 5.0 --z-down -10.0
```
- Presiona `g` para generar el G-code sin enviar
- Abre `regression_line.nc` y verifica que Z sube a 5 mm
- **Aumenta el valor** si necesitas más separación (ejemplo: 8, 10 mm)

### 2️⃣ **Calibración de Z-DOWN (escribir)**
```bash
python app.py --no-auto-send-grbl --z-up 5.0 --z-down 0.0
```
- Comienza con `0.0` (superficie)
- Si necesitas presión: usa valores **negativos** (-1.0, -2.0, -5.0)
- Si necesitas contacto suave: usa valores cerca de 0 (-0.5)

### 3️⃣ **Envío a GRBL con calibración**
```bash
python app.py --z-up 8.0 --z-down -1.5
```
- El código enviará automáticamente a GRBL
- Vigila que la herramienta:
  - ✓ Suba antes de mover XY
  - ✓ Baje para escribir
  - ✓ Vuelva a subir después

## Tips de calibración

🔧 **Para búsqueda y captura (probing):**
- Usa `--z-up 2.0` (poco levantamiento)
- Usa `--z-down -0.1` (muy suave)

✏️ **Para escritura o grabado:**
- Usa `--z-up 10.0` (levantamiento seguro)
- Usa `--z-down -2.0 a -5.0` (según dureza del material)

⚡ **Para corte:**
- Usa `--z-up 15.0` (separación amplia)
- Usa `--z-down -10.0 a -20.0` (presión variable según herramienta)

## Ejemplos de uso

**Opción 1: Valores por defecto**
```bash
python app.py
```

**Opción 2: Personalizado**
```bash
python app.py --z-up 7.5 --z-down -2.0 --feed-rate 400
```

**Opción 3: Solo generar, no enviar**
```bash
python app.py --z-up 5.0 --z-down 0.0 --no-auto-send-grbl
```

## Verificación visual

Después de presionar `g`:

```
✓ Generada gráfica: regression_plot.png
✓ Generado G-code: regression_line.nc
✓ Enviado a GRBL por COM3 (u otro puerto)
```

Verifica el archivo `regression_line.nc` con editor de texto si necesitas debug.
