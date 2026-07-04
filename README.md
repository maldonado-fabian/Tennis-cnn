# Detección de eventos de tenis a partir de audio

Proyecto del curso INF398 (Aprendizaje Automático).

## 1. Descripción del proyecto

Este proyecto detecta automáticamente eventos de un partido de tenis —golpes, saques y los
límites de cada punto— usando **únicamente la señal de audio**, sin procesar el video. El
problema se plantea como una clasificación por frame (a 25 fps) sobre características **MFCC + sus
derivadas (Δ, ΔΔ)**, resuelta con dos redes convolucionales 1D complementarias:

- **Modelo 1** — clasifica cada frame en `hit` / `serve` / `background`.
- **Modelo 2** — separa `in_play` (juego) de `between_points` (descanso).

Un post-procesamiento agrupa las predicciones por frame en eventos con marcas de tiempo y calcula
estadísticas del partido (golpes por rally, dobles faltas, intensidad, fatiga). El sistema se
valida contra las anotaciones reales de puntos (`points.txt`).

Todo el pipeline está en el notebook **`tennis_pipeline.ipynb`**. El informe y la presentación
(LaTeX) están en `informe/`.

```
tennis_pipeline.ipynb   Notebook principal (pipeline completo)
labels/                 Anotaciones por frame (V0XX.txt) y de puntos (points.txt)
informe/                Informe y presentación en LaTeX + figuras
requirements.txt        Dependencias de Python
```

## 2. Requisitos e instalación

**Software:**
- Python 3.10 o superior
- **ffmpeg** (para convertir audio/video a WAV) — `brew install ffmpeg` (macOS) o
  `apt install ffmpeg` (Linux)
- Paquetes de Python listados en `requirements.txt` (librosa, numpy, pandas, scikit-learn,
  torch, torchvision, matplotlib, soundfile)

**Instalación de dependencias:**

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Datos

Las **anotaciones** ya están incluidas en `labels/`. El **audio** de los 5 partidos
(`V006`–`V010`) es muy pesado (~1 GB en total, con archivos de hasta ~500 MB), por lo que
**no se versiona** en el repositorio (ver `.gitignore`).

**Descarga del audio:** los WAV están disponibles en la siguiente carpeta de Google Drive:

<https://drive.google.com/drive/folders/1yD9EwBeLz4t3tJtQeFGarCt4vB3fL8Hu?usp=sharing>

Descárgalos y colócalos en una carpeta `videos/` en la raíz del proyecto:

```
videos/V006.wav ... videos/V010.wav   (mono, 16 kHz)
```

Los datos provienen del conjunto asociado al artículo base *Detection of Tennis Events from
Acoustic Data* (MMSports 2019, DOI: 10.1145/3347318.3355520). Si tienes los partidos en video,
también puedes generar cada WAV con:

```bash
ffmpeg -i partido.mp4 -vn -ac 1 -ar 16000 -sample_fmt s16 videos/V0XX.wav
```

## 4. Ejecución y reproducción de los experimentos

Abre `tennis_pipeline.ipynb` y ejecútalo de arriba a abajo (**Restart & Run All**). El notebook
está dividido en secciones y reproduce, en orden, cada experimento del informe:

| Sección del notebook | Qué produce (resultado reproducible) |
|---|---|
| Datos + características | Alineación frame↔MFCC y el corpus `X`, `y` (Modelo 1) |
| Modelo 1 | Entrenamiento, **classification report** y matriz de confusión (Tabla/Fig. del Modelo 1) |
| Modelo 2 | Dataset `in_play`/`between_points`, entrenamiento y evaluación (Tabla/Fig. del Modelo 2) |
| Post-procesamiento | Eventos con timestamps y estadísticas del partido |
| Validación | `% de puntos con ≥1 evento` y métricas contra `points.txt` |
| Figuras | Exporta `curvas_m1`, `cm_m1`, `cm_m2`, `match_analysis` a `informe/figuras/` |

Para analizar un **audio/video nuevo** (sin anotaciones), usa la función incluida:

```python
events, stats, rallies, temporal = analyze_new_match("ruta/al/archivo.mp4")
```

> **Reproducción parcial sin el audio:** las celdas de entrenamiento y evaluación requieren los
> WAV en `videos/`. Si solo quieres inspeccionar el pipeline, las anotaciones de `labels/`
> permiten ejecutar las celdas de carga y de métricas basadas en `points.txt`.

## 5. Nota sobre el uso de IA

Parte del desarrollo de este proyecto (código, documentación y redacción) se apoyó en el
asistente de IA **Claude** (Anthropic). Las decisiones de diseño, la ejecución de los
experimentos y la revisión final de los resultados son responsabilidad del autor.

