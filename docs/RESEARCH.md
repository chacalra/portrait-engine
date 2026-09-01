# RESEARCH

Fecha: 2026-08-31. Autor: fase 1 del Portrait Retouching Engine.

Este documento recoge lo que se investigó antes de escribir código, qué se decidió
y por qué. Es la base de `docs/ARCHITECTURE.md` y `docs/ALGORITHMS.md`.

---

## 1. Resumen de decisiones

| Tema | Decisión | Motivo |
|---|---|---|
| Percepción principal | MediaPipe Face Landmarker (478 puntos) | Apache-2.0, sin restricción comercial, un solo modelo da bbox, landmarks densos e iris. Wheel `py3-none-manylinux` (funciona en Python 3.13). |
| Face parsing semántico | BiSeNet ONNX como backend **opcional y desactivado por defecto** | El código es MIT pero los pesos habituales se entrenan con CelebAMask-HQ, cuyo dataset es no comercial. Riesgo legal real (ver §5). |
| Máscaras | Derivadas de landmarks + modelo de probabilidad de piel por imagen + guided feathering | Cubre skin, ojos, cejas, labios, boca, dientes, nariz, contorno y under-eye sin modelos extra. |
| Filtro base edge-preserving | Guided filter propio (He et al. 2010, implementación en NumPy/`cv2.boxFilter`) | O(1) por píxel, sin halos, sin depender de `opencv-contrib`. |
| Restauración de textura | Separación de frecuencias en 3 bandas con guided filter | Consciente de bordes; DWT descartado (ver §4.3). |
| Espacio de trabajo | `float32` RGB sRGB-codificado en `[0,1]`, conversiones explícitas a Lab y a luz lineal | Evita pérdida acumulada por requantización uint8 entre etapas. |
| Modelos generativos | Ninguno | Requisito no negociable. |

---

## 2. Referencias estudiadas

### A. `axidex/face-retouching` (GPL-3.0)

Implementación no oficial de **FabSoften** (Velusamy, Parihar, Kini, Rege; CVPRW 2020).
Lo relevante de su arquitectura, según su documentación pública:

- Detección de cara con dlib HOG.
- Máscara binaria de piel con face parsing neuronal (excluye pelo, gorros, gafas, cuello,
  ojos, cejas, labios).
- Refinamiento de máscara: clustering GMM para obtener una posterior bayesiana de piel
  (`P_skin`), y **guided feathering** para convertirla en un alpha matte suave.
- Detección y ocultación de imperfecciones: DoG, Canny, inpainting.
- **Dynamic smoothing**: los hiperparámetros del filtro anisotrópico se calculan a partir de
  atributos medidos, con la forma `radius = 10 * P_skin + 10` y
  `epsilon = 5 * num_spots + 100`.
- **Texture restoration** mediante transformada wavelet discreta (DWT).
- Parámetros expuestos: `--strength` (multiplicador de epsilon), `--eps-cap` (300),
  `--max-radius` (20), `--texture-decay` (0.5), `--no-spots`.
- Salidas de debug: máscaras por cara, alpha mattes, mapas `P_skin`, máscaras de blemish
  y etapas intermedias.
- Usa pesos BiSeNet `resnet18.onnx` cargados por OpenCV DNN (requiere OpenCV >= 4.8).

**Influencia adoptada** (ideas y estructura del pipeline, no código):
1. La secuencia máscara aproximada -> probabilidad de piel -> guided feathering -> alpha matte.
2. Parámetros de suavizado dinámicos derivados de atributos medidos en lugar de constantes fijas.
3. La idea de suavizar y después restaurar textura, en lugar de suavizar menos.
4. La lista de salidas de debug.

**No adoptado**: DWT para restauración de textura (§4.3), GMM (§4.2), dlib.

### B. `georgegach/facestudio` (GPL-3.0-only)

Aplicación web que corre todo en el dispositivo. Lo útil como referencia conceptual:

- Percepción: MediaPipe Face Landmarker de 478 puntos para detección/landmarks, más
  BiSeNet para parsing semántico (skin, teeth, hair, etc.).
- Separación clara entre percepción (MediaPipe + BiSeNet) y operaciones (warp y filtros
  se aplican encima, en una etapa distinta).
- El parsing genera máscaras por componente facial y las herramientas editan sólo su región.
- Ejecución 100% cliente, modelos por WASM/WebGPU, ONNX Runtime Web para upscaling.

**Influencia adoptada**: confirma la combinación MediaPipe (landmarks densos) + parsing
opcional, y el patrón "percepción una vez, muchas operaciones".
**No adoptado**: su warp geométrico piecewise-affine en WebGL2. Nuestro motor prohíbe
modificar geometría facial.

### C. OpenCV G-API "face beautification" (Apache-2.0)

Es la referencia más importante porque codifica el paradigma que queremos: deep learning
para percepción, CV clásica para edición. Algoritmo exacto:

- Detección: `face-detection-adas-0001` (SSD, blob `[1x1x200x7]`).
- Landmarks: `facial-landmarks-35-adas-0002` (35 puntos: 18 elementos + 17 de mandíbula).
- Los 35 puntos no bastan para máscaras, así que **interpola contornos cerrados**:
  - Ojo: media elipse a partir de los dos cantos, con semieje `Y = 1/3` de la longitud del ojo,
    usando `cv::ellipse2Poly` y `atan2` para la inclinación (funciona con caras invertidas).
  - Frente: media elipse a partir de tres puntos de la mandíbula (izquierdo, derecho e inferior),
    asumiendo anchura de frente = anchura de mandíbula y altura de frente = `2/3` de la altura
    de mandíbula.
- Tres máscaras disjuntas:
  1. `sharp`: rellena los contornos de elementos faciales y se difumina con Gaussian.
  2. `blur`: rellena todo el contorno facial, difumina y **resta** la intersección con `sharp`.
  3. `background`: suma de las dos anteriores, umbral a 255 y `bitwise_not`.
- Composición final: `O = b*I + p*U + s*L`, con `U` = unsharp mask y `L` = bilateral.
- `unsharpMask(I, sigma, strength) = I - strength * Laplacian(medianBlur(I, sigma))`.
- Constantes reales del sample: kernel gaussiano de feathering `5x5` con `sigma=0`,
  bilateral `sigmaColor=30`, `sigmaSpace=30`, unsharp `sigma=3`.

**Influencia adoptada**:
1. La composición por máscaras disjuntas y ponderadas, que garantiza que no se toca nada
   fuera de región. Nuestro `apply_masked` es una versión de esto.
2. Difuminar la máscara en lugar de exigir contornos pixel-exactos.
3. Interpolación de contornos a partir de pocos puntos (aunque con 478 landmarks casi no
   la necesitamos; sí la usamos para la región infraorbital y la frente).

**No adoptado**: bilateral global sobre toda la piel como único suavizado, porque produce
el efecto plástico que el requisito 12 prohíbe. Tampoco el unsharp basado en Laplaciano de
mediana, que amplifica ruido; usamos realce de banda fina controlado.

### D. `WWIIITT/PortraitRetouch` (sin LICENSE declarada)

Sólo se estudió su arquitectura, no su enfoque de edición.

- `server.py` con rutas FastAPI por workflow (`/api/visual`, `/api/aesthetic`,
  `/api/style-transfer`, `/api/standard`).
- `src/agents/`: planificador de retoque, analizador estético, memoria de interacción.
- `src/processors/`: visual prompting (máscaras y crops), retoque facial local, corrección
  de color de piel corporal, parsing de formatos.
- Los processors hacen operaciones deterministas (resize, masking) y los modelos generativos
  se llaman aparte, vía `src/clients/gemini_client.py`.
- El router elige processor según endpoint; **no** hay registro dinámico de herramientas
  expuesto al agente.

**Influencia adoptada**: la separación agente / processors deterministas, y que FastAPI sea
sólo un adaptador. **Mejora deliberada sobre esta referencia**: nosotros sí exponemos un
registro de herramientas con contrato JSON (`portrait_engine.tools.registry`), que es lo que
el futuro agente necesita para tool calling.
**Rechazado**: la edición vía API generativa (Gemini). Nuestro motor no llama a ninguna API
externa.

---

## 3. Tecnologías evaluadas para la capa de percepción

| Opción | Licencia código | Licencia pesos | Da | Veredicto |
|---|---|---|---|---|
| **MediaPipe Face Landmarker** | Apache-2.0 | Apache-2.0 (model card Google) | bbox, 478 landmarks 3D, iris, matriz de transformación 4x4, blendshapes | **Elegido como base.** |
| MediaPipe Face Detector (BlazeFace) | Apache-2.0 | Apache-2.0 | bbox + 6 keypoints | Redundante: el Landmarker ya detecta. No se usa. |
| dlib 68 landmarks | Boost | shape_predictor: **sólo investigación** | 68 puntos | Descartado: menos puntos y pesos con restricción. |
| BiSeNet face parsing (`zllrunning/face-parsing.PyTorch`) | MIT | entrenado con CelebAMask-HQ (**no comercial**) | 19 clases incl. hair, mouth, u_lip, l_lip | Backend opcional, off por defecto. |
| OpenVINO `facial-landmarks-35-adas-0002` | Apache-2.0 | Apache-2.0 | 35 puntos | Descartado: menos denso que MediaPipe y añade dependencia de OpenVINO. |
| InsightFace / SCRFD | MIT | pesos **sólo investigación** | detección robusta | Descartado por licencia de pesos. |
| ONNX Runtime | MIT | n/a | inferencia | Se mantiene como dependencia opcional para el backend de parsing y para la migración futura. |

### Por qué 478 landmarks bastan

El sample de OpenCV construye máscaras usables con **35** puntos interpolando elipses. Con 478
tenemos anillos cerrados y ordenados para: contorno facial (36 puntos), cada ojo (16), cada iris
(4), labio externo (20), labio interno (20), cejas (10 por lado) y nariz. Es decir, la
interpolación agresiva de OpenCV deja de ser necesaria salvo en dos casos:

1. **Región infraorbital (ojeras)**: no existe como anillo. Se deriva desplazando el párpado
   inferior hacia la mejilla (ver `docs/ALGORITHMS.md`).
2. **Frente por encima de la línea del pelo**: el óvalo facial de MediaPipe sigue
   aproximadamente la línea del pelo, y en algunos peinados incluye pelo. Se resuelve con la
   probabilidad de piel, no con geometría.

Lo que los landmarks **no** dan y sí daría BiSeNet: pelo, cuello, gafas, ropa y una separación
limpia de dientes. Para el MVP se sustituye con:
- pelo/barba: máscara de "textura tipo pelo" basada en energía de alta frecuencia local y
  luminancia baja, calculada en la propia imagen;
- dientes: interior del labio interno filtrado por luminancia alta y cromaticidad baja.

Riesgo asumido y documentado. Si la calidad no basta, el backend BiSeNet ya tiene su interfaz.

---

## 4. Algoritmos: comparación y elección

### 4.1 Suavizado edge-preserving

| Técnica | Coste | Halos | Control de banda | Notas |
|---|---|---|---|---|
| Gaussian blur | bajo | sí | ninguno | Prohibido por el requisito: destruye textura y bordes. |
| Bilateral filter | alto (O(r^2) sin aproximar) | bajos | ninguno | Es lo que usa OpenCV G-API. Tiende a "plastificar" con sigma alto porque aplana también la banda fina. |
| **Guided filter** (He, Sun, Tang, ECCV 2010) | **O(1) por píxel** | **no** (por construcción es un filtro lineal local) | indirecto vía radio | Elegido. Además sirve como matting/feathering guiado por la imagen, que es exactamente el "guided feathering" de FabSoften. |
| Domain transform / L0 / WLS | medio-alto | bajos | no | Sin ventaja clara sobre guided para este caso. |
| Anisotropic diffusion (ADF) | iterativo, medio | bajos | no | Lo que usa FabSoften. Más lento y con más parámetros que guided para un resultado equivalente aquí. |

**Decisión**: guided filter como primitiva única de suavizado y de feathering. Un solo bloque
bien testeado sirve para skin smoothing, ojeras, iluminación y matting.

### 4.2 Probabilidad de piel

FabSoften usa GMM para una posterior bayesiana. Alternativas:

- Umbrales fijos en YCrCb o HSV (literatura clásica de detección de piel): frágiles ante
  iluminación y tono de piel.
- GMM global entrenado offline: requiere dataset y no se adapta a la foto.
- **Modelo gaussiano por imagen, sembrado desde la propia cara** (elegido): se toman parches
  de mejilla y frente que están garantizados dentro del óvalo y lejos de rasgos, se estima
  media y covarianza en `(L*, a*, b*)`, y se calcula `P_skin = exp(-0.5 * d_mahalanobis^2)`.

Es más simple que un GMM, no necesita pesos, y se adapta automáticamente al tono de piel y a
la iluminación de esa foto concreta. Es la razón de que el motor funcione igual en pieles
claras y oscuras sin recalibrar. Coste: una covarianza 3x3.

### 4.3 Restauración de textura: DWT frente a separación de frecuencias con guided filter

FabSoften restaura textura manipulando bandas wavelet. Se descartó DWT por tres razones:

1. La DWT no es consciente de bordes. Reinyectar bandas wavelet cerca de una ceja o de un
   párpado reintroduce ringing, precisamente el artefacto que el criterio de calidad prohíbe.
2. Depende del tamaño de la imagen (padding a potencias de dos) y del ROI, lo que hace el
   resultado menos reproducible al cambiar el crop.
3. Una separación en tres bandas con dos guided filters de radio distinto da el mismo control
   (base / mid / fine) con bordes respetados, coste O(n) y dos parámetros interpretables.

**Elegido**: separación en 3 bandas. Se atenúa la banda **media** (manchas, irregularidad de
tono) y se preserva íntegra la banda **fina** (poros). Esto es lo que hace que el resultado
parezca retoque y no filtro beauty: el poro sigue ahí.

`docs/ALGORITHMS.md` tiene las fórmulas.

### 4.4 Blemish healing

- `cv2.inpaint` (Telea o Navier-Stokes) directo: rellena con contenido suave. Elimina el
  granito pero también el poro, y deja un parche liso detectable.
- Clonado por parches (PatchMatch): visible como repetición de textura.
- **Healing por separación de frecuencias** (elegido): se inpaintea solamente la banda **baja**
  y se reinyecta la banda **fina original** del propio píxel. El granito desaparece porque su
  energía vive en la banda baja/media local, y la textura de piel se conserva porque nunca se
  toca. Es lo que hace un retocador con "Healing Brush" en modo frecuencia separada.

### 4.5 Blanqueo de dientes

Trabajar en Lab y **multiplicar** `b*` por un factor menor que 1 en lugar de restar un offset.
Multiplicar preserva la estructura de sombras (una zona con `b*` bajo se mueve poco) y evita
el blanco plano. La subida de `L*` se hace proporcional al espacio libre `(L_cap - L)`, lo que
impide llegar a blanco puro por construcción.

### 4.6 Ojeras

Clave: corregir **solamente la componente de baja frecuencia del déficit de luminancia**
respecto a la mejilla vecina. Así la bolsa, el pliegue y las arrugas (alta frecuencia)
sobreviven, y lo que se reduce es la mancha. Corrección limitada a una fracción del déficit
(60% como máximo a strength 1.0), por lo que es reducción y no borrado.

### 4.7 Iluminación facial

Estimar la iluminación como componente de baja frecuencia de la luminancia **en luz lineal** y
aplicar una ganancia `(target / I_low)^(k*strength)` acotada. En luz lineal la ganancia es
fotométricamente correcta y no vira el color. El exponente acotado y el clip de ganancia son
lo que evita el aspecto HDR. Los altas luces especulares se comprimen con un rolloff suave en
lugar de recortarse.

---

## 5. Licencias y riesgos

Detalle completo en `docs/THIRD_PARTY.md`. Lo crítico:

| Riesgo | Severidad | Mitigación |
|---|---|---|
| `axidex/face-retouching` y `facestudio` son GPL-3.0 | Alta si se copiara código | No se ha copiado ni una línea. Sólo ideas y algoritmos publicados (paper CVPRW 2020 y documentación). Toda la implementación es propia. Documentado aquí y en `THIRD_PARTY.md`. |
| Pesos BiSeNet entrenados con CelebAMask-HQ (dataset no comercial) | Alta para producto comercial | Backend opcional, desactivado por defecto, sin pesos incluidos en el repo, y aviso explícito al activarlo. |
| Pesos MediaPipe Face Landmarker | Baja | Model card de Google bajo Apache-2.0. **No verificado por asesoría legal**; conviene confirmarlo antes de comercializar. |
| `PortraitRetouch` sin LICENSE | Media | Sin licencia declarada equivale a "todos los derechos reservados". No se ha reutilizado nada, sólo se describió su arquitectura públicamente visible. |
| Imágenes de prueba (Wikimedia, CC BY / CC BY-SA) | Baja | Viven en `data/`, git-ignored, con `ATTRIBUTION.md`. No se redistribuyen con el paquete. |

Declaración de incertidumbre: las afirmaciones sobre licencias de pesos vienen de la
documentación pública de cada proyecto y de búsquedas web, no de una revisión legal. Antes de
usar esto en un producto comercial hay que verificarlas de nuevo.

---

## 6. Riesgos técnicos identificados

1. **MediaPipe no detecta todas las caras.** En la batería de prueba, `old_man_imagicity_231.jpg`
   devuelve 0 caras (la cara es pequeña respecto al encuadre). Mitigación en el MVP: error
   controlado `NoFaceDetectedError`. Mejora futura: detección en dos pasos (detector, crop, luego
   landmarker sobre el crop).
2. **`mediapipe.solutions` ya no existe en MediaPipe 1.x.** Las constantes `FACEMESH_*` no se
   pueden importar. Los índices se fijan en `portrait_engine/perception/landmark_sets.py`,
   derivados de `face_mesh_connections.py` (Apache-2.0) y con su origen documentado.
3. **Línea del pelo y barba.** Los landmarks no distinguen piel de pelo. Se mitiga con
   probabilidad de piel y máscara de textura tipo pelo. Es la principal debilidad conocida del MVP.
4. **Dientes sin parsing semántico.** La detección por luminancia dentro del labio interno falla
   con boca cerrada o poco abierta. Se mitiga devolviendo la operación como no-op cuando el área
   detectada es demasiado pequeña, en lugar de blanquear la lengua.
5. **Pérdida acumulada de precisión.** Cinco herramientas en cadena sobre uint8 degradan.
   Mitigado por el buffer `float32` único de la pipeline.
6. **Reproducibilidad.** Nada del pipeline usa aleatoriedad. MediaPipe es determinista para el
   mismo binario y la misma imagen. Se testea igualdad bit a bit entre dos ejecuciones.

---

## 7. Fuentes

- Velusamy, Parihar, Kini, Rege. *FabSoften: Face Beautification via Dynamic Skin Smoothing,
  Guided Feathering, and Texture Restoration*. CVPRW 2020.
  <https://ieeexplore.ieee.org/document/9150825/>
- He, Sun, Tang. *Guided Image Filtering*. ECCV 2010 / TPAMI 2013.
- Tomasi, Manduchi. *Bilateral Filtering for Gray and Color Images*. ICCV 1998.
- Telea. *An Image Inpainting Technique Based on the Fast Marching Method*. 2004.
- OpenCV. *Implementing a face beautification algorithm with G-API*.
  <https://docs.opencv.org/4.x/tutorials_contrib/gapi/face_beautification/face_beautification.html>
  (código: `samples/cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp`)
- MediaPipe Face Landmarker. <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker>
- `axidex/face-retouching`. <https://github.com/axidex/face-retouching>
- `georgegach/facestudio`. <https://github.com/georgegach/facestudio>
- `WWIIITT/PortraitRetouch`. <https://github.com/WWIIITT/PortraitRetouch>
- `zllrunning/face-parsing.PyTorch`. <https://github.com/zllrunning/face-parsing.PyTorch>
