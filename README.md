# Generador de Archivos Parquet Personalizado

Este script permite generar archivos Parquet con datos sintéticos de manera eficiente y personalizada, ideal para pruebas de rendimiento, desarrollo o demostraciones.

## Características

- Generación de datos sintéticos con diferentes motores (mimesis, faker, random)
- Procesamiento en paralelo para mayor rendimiento
- Soporte para generación por lotes
- Optimizaciones de caché para mejorar el rendimiento
- Estadísticas detalladas de tiempos de generación
- Diferentes opciones de compresión

## Requisitos

- Python 3.6+
- Dependencias:
  - pyarrow
  - pandas
  - numpy
  - mimesis (opcional)
  - faker (opcional)

## Instalación de dependencias

```bash
pip install pyarrow pandas numpy
pip install mimesis faker  # Opcional, pero recomendado para datos más realistas
```

## Uso básico

```bash
python generate_parquet_custom.py --rows 100000 --columns 10 --filename datos.parquet
```

## Argumentos disponibles

| Argumento | Descripción | Valor predeterminado | Valores permitidos |
|-----------|-------------|----------------------|-------------------|
| `--rows` | Cantidad total de filas a generar | 10000 | Cualquier entero positivo |
| `--columns` | Cantidad de columnas a generar | 5 | Cualquier entero positivo |
| `--batch-size` | Número de filas por lote | 1000 | Cualquier entero positivo |
| `--filename` | Nombre base del archivo de salida | "output.parquet" | Cualquier nombre de archivo válido |
| `--threads` | Cantidad de procesos a utilizar | 0 (autodetectar) | Entero (0 = auto, o número específico) |
| `--output-dir` | Ruta de salida para el archivo | "" (carpeta actual) | Ruta válida existente |
| `--mode` | Modo de generación de datos | "mimesis" | "mimesis", "faker", "random" |
| `--compression` | Algoritmo de compresión | "snappy" | "none", "snappy", "gzip", "brotli", "zstd" |
| `--column-parallel` | Usar paralelización por columnas | False (sin bandera) | Bandera booleana (presente o no) |
| `--cache-size` | Tamaño máximo de caché para valores frecuentes | 1000 | Entero (0 o mayor) |

## Detalles de los modos de generación

### Modo "mimesis"
- Genera datos realistas en español usando la biblioteca mimesis
- Alterna columnas entre nombres completos y direcciones
- Más rápido que faker, pero requiere la biblioteca mimesis instalada
- Si mimesis no está disponible, se usará faker automáticamente

### Modo "faker"
- Genera datos realistas en español usando la biblioteca faker
- Alterna columnas entre nombres y direcciones
- Más variedad que mimesis, pero un poco más lento
- Si faker no está disponible, se usará generación aleatoria básica

### Modo "random"
- Genera cadenas aleatorias de caracteres alfanuméricos
- Es el modo más rápido, pero los datos no son realistas
- No requiere bibliotecas adicionales
- Se usa como fallback si mimesis y faker no están disponibles

## Optimizaciones

### Paralelización por columnas (`--column-parallel`)
- Genera cada columna en paralelo usando múltiples procesos
- Aumenta significativamente el rendimiento en sistemas con múltiples núcleos
- Usa más memoria que la generación secuencial

### Caché de valores (`--cache-size`)
- Reutiliza valores generados para simular distribuciones de datos más realistas
- 70% de probabilidad de usar un valor cacheado
- Mejora el rendimiento y simula datos del mundo real con repeticiones
- Establecer a 0 para desactivar la caché

### Comprensión (`--compression`)
- "snappy": Buena relación entre velocidad y compresión (predeterminado)
- "gzip": Mayor compresión, pero más lento
- "brotli": Muy buena compresión, más lento
- "zstd": Excelente equilibrio entre velocidad y compresión
- "none": Sin compresión, archivos más grandes pero generación más rápida

## Ejemplos de uso

### Generación básica
```bash
python generate_parquet_custom.py --rows 10000 --columns 5
```

### Generación de archivo grande con máximo rendimiento
```bash
python generate_parquet_custom.py --rows 1000000 --columns 20 --batch-size 50000 --mode random --compression none --column-parallel
```

### Generación con máxima compresión
```bash
python generate_parquet_custom.py --rows 100000 --columns 10 --compression gzip
```

### Generación con datos realistas
```bash
python generate_parquet_custom.py --rows 50000 --columns 15 --mode faker --column-parallel --cache-size 5000
```

## Salida de información

El script muestra información detallada durante la ejecución:
- Opciones elegidas
- Generador de datos utilizado
- Optimizaciones activas
- Progreso por lotes con tiempos
- Resumen estadístico final con tiempos mínimos, máximos y promedio
- Ruta del archivo Parquet generado

## Ayuda integrada

Para ver todas las opciones disponibles:
```bash
python generate_parquet_custom.py --help
```
