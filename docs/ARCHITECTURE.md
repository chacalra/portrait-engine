# ARCHITECTURE

Portrait Retouching Engine. Fase 2.

El motor es una **librería Python de procesamiento de imagen**. No es un servicio, no tiene
estado persistente y no llama a ninguna API externa. FastAPI es un adaptador opcional.

---

## 1. Principio estructural

```
        percepción (una vez)                 edición (N veces, independiente)
   ┌──────────────────────────┐        ┌────────────────────────────────────┐
   │ FaceAnalyzer             │        │ tool(image, analysis, strength)    │
   │  landmarks + parsing opc │──────▶ │  -> ToolResult(image, stats, dbg)  │
   │  -> FaceAnalysis         │        └────────────────────────────────────┘
   └──────────────────────────┘                        ▲
                                                       │
                                         PortraitEngine.process() orquesta
```

Tres reglas que definen la arquitectura:

1. **La percepción no sabe nada de las herramientas.** Produce un `FaceAnalysis` inmutable.
2. **Las herramientas no ejecutan modelos.** Reciben `FaceAnalysis` y sólo hacen aritmética
   de imagen. Son funciones puras.
3. **Todo píxel escrito pasa por `apply_masked`.** Es el único punto del código que compone
   original y editado, y es donde viven los guardrails de identidad.

## 2. Layout

```
portrait_engine/
├── core/
│   ├── types.py          FaceAnalysis, BBox, MaskSet, ToolResult, RetouchResult, errores
│   ├── colorspace.py     sRGB<->linear, RGB<->Lab/YCrCb, luma, gestión de precisión
│   ├── image.py          PortraitImage (float32 RGB + metadata), load/save, EXIF/ICC
│   ├── ops.py            guided filter, feathering, separación de bandas, curvas, blend
│   ├── masking.py        apply_masked + verificación de no-escritura fuera de máscara
│   ├── config.py         constantes por herramienta y mapeo strength -> parámetros
│   └── pipeline.py       PortraitEngine
├── perception/
│   ├── landmark_sets.py  índices de FaceMesh (anillos ordenados)
│   ├── landmarks.py      backend MediaPipe -> LandmarkResult
│   ├── face_parsing.py   backend BiSeNet ONNX opcional (off por defecto)
│   ├── masks.py          construcción de máscaras y probabilidad de piel
│   └── face_analyzer.py  FaceAnalyzer
├── tools/
│   ├── base.py           validación de strength, registro, contrato JSON
│   ├── teeth_whitening.py
│   ├── skin_smoothing.py
│   ├── dark_circles.py
│   ├── face_lighting.py
│   └── skin_cleanup.py
├── debug/
│   ├── visualize.py      landmarks, bbox, overlays de máscara, before/after, difference map
│   └── recorder.py       DebugRecorder: escribe artefactos numerados a disco
├── diagnostics.py        métricas de preservación de identidad
├── cli.py                portrait-retouch
└── api/app.py            FastAPI: POST /analyze, POST /retouch (extra opcional)
```

## 3. Tipos centrales

### `PortraitImage`

Buffer canónico. **`float32`, orden RGB, sRGB-codificado, rango `[0,1]`, shape `(H, W, 3)`.**

Decisiones y su motivo:

- **RGB y no BGR internamente.** OpenCV usa BGR; la conversión ocurre sólo en `load`/`save`.
  Una sola frontera evita la clase de bug más común de este dominio.
- **`float32` y no `uint8`.** Cinco herramientas en cadena sobre `uint8` acumulan error de
  cuantización visible en degradados de piel. El buffer nunca se requantiza entre etapas.
- **sRGB-codificado y no lineal como representación base.** La mayoría de las operaciones
  (Lab, cromaticidad, detección de manchas) son perceptuales y quieren sRGB. Las operaciones
  fotométricas (exposición, iluminación) convierten explícitamente a luz lineal, operan y
  vuelven. La conversión es explícita en el sitio de uso, nunca implícita.
- La metadata (ruta, dtype original, EXIF, ICC) viaja aparte en `ImageMeta`, no dentro del array.

### `FaceAnalysis` (inmutable)

```python
FaceAnalysis(
    image_shape=(H, W),
    bbox=BBox(x, y, w, h),
    landmarks=np.ndarray,          # (478, 2) float32, píxeles
    landmarks_3d=np.ndarray,       # (478, 3) float32, z normalizado (no se usa para editar)
    masks=MaskSet({...}),          # float32 [0,1], HxW, ya feathered
    skin_probability=np.ndarray,   # float32 [0,1], HxW
    geometry=FaceGeometry(interocular_px, face_width_px, roll_degrees, ...),
)
```

`masks` contiene, como mínimo: `skin`, `face_oval`, `left_eye`, `right_eye`, `eyes`,
`left_eyebrow`, `right_eyebrow`, `eyebrows`, `lips`, `mouth_interior`, `teeth`, `nose`,
`nostrils`, `hair_like`, `left_under_eye`, `right_under_eye`, `under_eye`, `protected`.

`protected` es la unión de lo que ninguna herramienta de piel puede tocar (ojos, cejas,
labios, fosas nasales, textura tipo pelo). Existe para que el guardrail sea una sola
comprobación y no una lista repetida en cada herramienta.

`geometry.interocular_px` es la unidad de escala del motor: **todos** los radios de filtro,
tamaños de blob y offsets se expresan como fracción de esa distancia. Es lo que hace que el
mismo `strength` produzca el mismo resultado visual en una foto de 800 px y en una de 6000 px.

### `ToolResult` y `RetouchResult`

```python
ToolResult(image, tool, params, stats, changed)     # stats: área editada, deltas medidos
RetouchResult(image, analysis, stages, diagnostics, debug_dir)
```

## 4. Máscaras: contrato

- Siempre `float32` en `[0, 1]`, siempre `(H, W)` con `H, W` de la imagen completa.
- Siempre feathered antes de salir de la capa de percepción. Las herramientas pueden
  restringirlas más (multiplicar), nunca expandirlas.
- Nunca binarias en el punto de composición. Un borde duro es un artefacto visible.

El feathering usa guided filter con la imagen como guía, no Gaussian. Esto es "guided
feathering": el borde de la máscara se pega a los bordes reales de la imagen, así que la
máscara de piel no se derrama sobre una ceja aunque el polígono sea impreciso.

## 5. Flujo de datos de `PortraitEngine.process`

```
path/array
   │  load: decodifica, aplica orientación EXIF, uint8 -> float32 RGB, guarda metadata
   ▼
PortraitImage
   │  FaceAnalyzer.analyze(): 1 inferencia MediaPipe, N máscaras, P_skin
   ▼
FaceAnalysis  ──────────────────────────────────┐  (compartido, sin recomputar)
   │                                            │
   ▼                                            │
skin_cleanup    (strength) ──▶ apply_masked ────┤
skin_smoothing  (strength) ──▶ apply_masked ────┤   cada etapa: guardrail
reduce_dark_circles         ──▶ apply_masked ───┤   + snapshot debug
whiten_teeth                ──▶ apply_masked ───┤
balance_face_lighting       ──▶ apply_masked ───┘
   │
   ▼
diagnostics: deriva landmarks, retención de textura, fuga fuera de máscara
   │
   ▼
save: float32 -> uint8 con dither opcional, reinyecta EXIF/ICC
```

**Orden por defecto y su razón**: limpiar antes de suavizar (si no, el granito se difumina y
se convierte en una mancha ancha más difícil de quitar). Ojeras después de suavizar (opera
sobre baja frecuencia ya estable). Dientes en cualquier punto (región disjunta). Iluminación
al final, porque es global y debe juzgar el resultado ya limpio.

El orden es un dato (`core/config.DEFAULT_ORDER`), no código, para que el futuro agente pueda
proponer otro.

## 6. Guardrails de identidad

Implementados, no sólo documentados:

| Guardrail | Dónde | Qué hace |
|---|---|---|
| Sin warping geométrico | `core/masking.apply_masked` | Sólo acepta `edited` con el mismo shape y compone por álgebra por píxel. Ninguna herramienta importa `warpAffine`, `remap` ni `seamlessClone`. Un test lo verifica con AST. |
| Sin escritura fuera de máscara | `apply_masked(strict=True)` | Calcula el máximo delta donde `mask == 0` y lanza si supera la tolerancia. |
| Landmarks intactos | `diagnostics.landmark_drift` | Re-ejecuta el landmarker sobre la salida y mide desplazamiento RMS normalizado por distancia interocular. Umbral de aviso configurable. |
| Textura preservada | `diagnostics.texture_retention` | Ratio de energía de alta frecuencia en la piel después/antes. Por debajo de ~0.55 el resultado es plástico. |
| Regiones protegidas | `MaskSet["protected"]` | Ojos, cejas, labios, fosas nasales y textura tipo pelo se restan de toda máscara de piel. |
| Techo de intensidad | `core/config.py` | `strength=1.0` mapea a un máximo definido por herramienta, elegido para seguir siendo natural. |

## 7. Preparación para el agente (Mundo 2)

`portrait_engine.tools.base.register` construye un registro con contrato JSON por herramienta:

```python
from portrait_engine.tools import TOOL_REGISTRY, tool_schemas
tool_schemas()   # [{"name": ..., "description": ..., "parameters": {...}}, ...]
```

El contrato se genera desde la firma y el docstring de la propia función, así que no puede
desincronizarse del código. El agente no se implementa aquí.

## 8. Preparación para migración (ONNX / C++ / Core ML)

- Ninguna herramienta usa aleatoriedad, bucles Python por píxel ni estructuras de datos
  dinámicas. Todo son operaciones de array y llamadas de OpenCV que tienen equivalente 1:1 en
  C++ y en Accelerate/Metal.
- El guided filter está implementado sobre `boxFilter`, que es la primitiva más portable.
- El único modelo es MediaPipe, que ya es TFLite y tiene runtime en iOS/Android.
- El backend de parsing habla ONNX Runtime.
- No se hace esta migración ahora.

## 9. Lo que deliberadamente no está

Sin base de datos, sin autenticación, sin cola de trabajos, sin caché en disco, sin
multiproceso, sin GPU, sin frontend. Sin soporte multi-cara (error controlado). Sin vídeo.
