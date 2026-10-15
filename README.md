# Portrait Retouching Engine

A programmable image-processing SDK for professional portrait retouching, built around deterministic edits and explicit identity-preservation guardrails.

**AI decides what to edit. Deterministic image tools perform the edit.**

Deep learning is used only to *perceive* the face. Every pixel that changes is produced by a
classical, parameterized, reproducible operation on the original photograph. No generative
model is involved, and facial geometry is never modified.

This repository is the tools half of that split. The agent that decides which operations to
apply is a separate project and is not here.

The portfolio value of this project is not just the visual result: the implementation separates perception from editing, keeps tool behavior reproducible, measures identity-preservation constraints, exposes the engine through CLI/library/HTTP interfaces, and verifies behavior in CI.

---

## What it does

Six capabilities, chosen to be good rather than numerous:

| tool | what it does | what it will not do |
|---|---|---|
| `skin_cleanup` | reduces small temporary imperfections by repairing only the smooth surface under each spot and reinjecting the original texture | remove moles or freckles, destroy pores, clone-stamp |
| `skin_smoothing` | attenuates the mid-frequency band of skin luminance and the unevenness of chroma | touch the fine band, so pores survive at any strength |
| `equalize_skin_tone` | shifts the low-frequency colour field toward the face's own median, correcting a ruddy cheek or a sallow temple | touch `L*`, so modelling, texture and facial form are untouched by construction |
| `reduce_dark_circles` | corrects the low-frequency luminance and color deficit under the eye, relative to the neighbouring cheek | flatten the eye area, erase the eye bag or the creases |
| `whiten_teeth` | reduces the yellow cast multiplicatively and lifts luminance into the remaining headroom | reach pure white, touch lips or gums, flatten inter-tooth shading |
| `balance_face_lighting` | applies a bounded, smooth photometric gain in linear light | relight generatively, produce an HDR look, clip highlights |

Every tool takes a `strength` in `[0, 1]`, where **`1.0` is the maximum this engine considers
natural**, not the maximum the arithmetic allows.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# optional extras
pip install -e ".[api]"      # FastAPI adapter
pip install -e ".[dev]"      # tests and lint
```

Python 3.10 or newer. The MediaPipe Face Landmarker model (3.7 MB, Apache-2.0) downloads
automatically into `assets/models/` on first use.

## Use

### Command line

```bash
portrait-retouch input.jpg \
  --skin-cleanup 0.15 \
  --skin-smoothing 0.20 \
  --dark-circles 0.15 \
  --teeth-whitening 0.10 \
  --face-lighting 0.10 \
  --debug
```

```
output/
  final.jpg
  debug/
    perception/  00_bbox.jpg  01_landmarks.jpg  02_mask_overview.jpg
    masks/       skin.png  teeth.png  left_under_eye.png  skin_probability.png  ...
    stages/      01_original.jpg  02_skin_cleanup.jpg  ...  99_final.jpg
    diffs/       02_skin_cleanup_diff.jpg  ...
    before_after.jpg  difference_map.jpg  difference_signed.jpg  report.json
```

`--preset natural` applies the strengths above. `--json` prints the full report.

### As a library

```python
from portrait_engine import PortraitEngine

engine = PortraitEngine()
result = engine.process(
    "portrait.jpg",
    skin_cleanup=0.15,
    skin_smoothing=0.20,
    dark_circles=0.15,
    teeth_whitening=0.10,
    face_lighting=0.10,
)
engine.save(result, "output/final.jpg")

print(result.diagnostics["texture_retention"])
print(result.diagnostics["warnings"])
```

### One tool at a time

Tools are independent pure functions. They never run a model; they take a `FaceAnalysis`.

```python
from portrait_engine import FaceAnalyzer, load_image, whiten_teeth

image = load_image("portrait.jpg")
analysis = FaceAnalyzer().analyze(image)     # one inference, reusable

result = whiten_teeth(image, analysis, strength=0.15)
print(result.changed, result.stats["mean_b_before"], result.stats["mean_b_after"])
```

### HTTP

```bash
pip install -e ".[api]"
uvicorn portrait_engine.api.app:app
```

`POST /analyze`, `POST /retouch`, `GET /tools`, `GET /health`. The adapter is 200 lines and
contains no logic: deleting it would not affect the engine.

## How it is built

```
perception (once)                          tools (independent)
┌────────────────────────────┐            ┌──────────────────────────────────┐
│ FaceAnalyzer               │            │ tool(image, analysis, strength)  │
│  MediaPipe: 478 landmarks  │───────────▶│  -> ToolResult(image, stats)     │
│  + per-image skin model    │            └──────────────────────────────────┘
│  + derived masks           │                          ▲
│  -> FaceAnalysis           │            PortraitEngine.process() orchestrates
└────────────────────────────┘
```

Three rules define the architecture:

1. Perception knows nothing about the tools. It produces an immutable `FaceAnalysis`.
2. Tools run no models. They are pure functions over arrays.
3. **Every pixel written passes through `apply_masked`.** That is the one place original and
   edited pixels are combined, and it is where the identity guardrails live.

Details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Formulas, parameter meanings and the
measurements behind every constant in [docs/ALGORITHMS.md](docs/ALGORITHMS.md). What was studied
and why each choice was made in [docs/RESEARCH.md](docs/RESEARCH.md).

## Identity preservation, as numbers

The engine measures itself. Every run reports:

| metric | what it catches | measured at strength 1.0 |
|---|---|---|
| out-of-region leak | any pixel changed outside the permitted masks | 0.02 to 0.04 of one 8-bit level |
| texture retention | plastic skin | 0.93 to 1.04 |
| landmark drift | changed facial geometry | 0.006 to 0.011 of the interocular distance, against a measured detector noise floor of 0.004 to 0.013 |

Structurally, no tool *can* warp geometry: `apply_masked` only composes per pixel, and a test
fails if any tool module so much as references `warpAffine`, `remap`, `resize`, `inpaint` or a
dozen similar functions. See [docs/ALGORITHMS.md](docs/ALGORITHMS.md) section 6.

## Ready for the agent

Each tool exposes a JSON contract generated from its own signature and docstring, so it cannot
drift from the code.

```python
from portrait_engine import tool_schemas
tool_schemas()
```

```json
{
  "name": "reduce_dark_circles",
  "description": "Reduce the visual intensity of under-eye dark circles while preserving the natural anatomical shadow, by correcting only the low-frequency luminance and color deficit relative to the neighbouring cheek.",
  "parameters": {
    "type": "object",
    "properties": {
      "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.2}
    }
  }
}
```

Also served at `GET /tools`. No agent code lives in this repository.

## Quality and reproducibility

GitHub Actions runs Ruff plus the non-network/non-slow test suite on Ubuntu. The workflow also installs the system EGL/OpenGL/GLES libraries required by MediaPipe, so the CI environment exercises the same perception stack rather than silently skipping it for missing native libraries.

**On a fresh clone, fetch the fixtures first.** No evaluation portraits are tracked in this repository, so
without this step the tests that need a real photograph skip rather than run.

```bash
python scripts/fetch_test_images.py     # CC-licensed evaluation portraits into data/input/

pytest                                  # 229 tests, about 36 seconds
pytest -m "not slow"                    # skip the ones needing real photos
ruff check portrait_engine scripts tests

python scripts/evaluate_folder.py data/input data/output/eval --strength-sweep
```

`evaluate_folder.py` processes a folder and writes, per image, a before/after comparison, a
difference map and an optional strength sweep, plus `evaluation.csv` with the numeric
diagnostics. That is the tool for judging a parameter change across a set of portraits rather
than one favourite photo.

The repository intentionally does not commit portrait fixtures or generated before/after images. This avoids redistributing evaluation photos and keeps the public repository focused on the engine, tests, diagnostics, and reproducible evaluation tooling.

## Scope and known limits

Stated plainly, because the failure modes are the interesting part.

**In scope for this version.** One person, one clearly visible frontal or semi-frontal face,
a real photograph, JPG or PNG, at a reasonable resolution.

**Rejected with a clear error.** No face detected. More than one face. A face smaller than 6
percent of the frame's long side. An undecodable or truncated file.

**Known weaknesses, measured:**

- **Hair, beard and neck separation is the weakest part of perception.** There is no semantic
  segmentation by default, so hair is inferred from an anatomical prior (the beard zone and the
  oval rim) plus relative local texture. A clean-shaven jaw in deep shadow is partly excluded
  from the skin mask, which costs a little smoothing coverage in a region that has nothing to
  smooth.
- **Blemish detection recall depends on the skin.** Measured with synthetic 10 `L*` spots: 6 of 6
  on smooth young skin, 3 of 6 on heavily freckled skin, 0 of 6 on dark low-contrast skin. The
  detector normalizes by local texture, so on very textured skin a mild spot genuinely is not
  unusual. Lowering the threshold to catch those cases would start removing real skin.
- **Blemish detection needs a reasonably large face**, roughly 150 px of interocular distance or
  more. Below that a spot is a couple of pixels across and is not separable from texture.
- **Teeth detection needs an open mouth.** With a closed or barely open mouth, `whiten_teeth`
  reports a no-op rather than guessing, which is the intended behaviour.
- **The single-face policy is strict.** A group photo errors instead of retouching the largest
  face, because silently retouching one of several people is a worse failure than an error.
- **MediaPipe misses some faces**, in particular small or strongly profile ones. One of the six
  evaluation fixtures is rejected for this reason. A two-stage detect-then-crop pass would fix
  it and is not implemented.
- **Model weight licensing is not legally verified.** See [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).

**Deliberately not built.** The agent. A mobile or web UI. Multi-face support. Video. Any
generative model. Geometric edits of any kind: no face slimming, no nose changes, no eye
enlargement, no lip changes, no age changes, no makeup, no hair, no background.

## Performance

About 1.4 to 2.1 seconds for a full five-tool run at 1400 px on a CPU, of which roughly a third
is the single MediaPipe inference. Not optimized: the priorities for this version were visual
quality, clarity, modularity and reproducibility, in that order. Nothing in the design blocks a
later port: no randomness, no Python per-pixel loops, one TFLite model, and the guided filter is
built on `boxFilter`, which has a direct equivalent in C++, Accelerate and Metal.

## License

Proprietary. Third-party licenses and obligations in
[docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).
