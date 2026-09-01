# THIRD PARTY

Every external dependency, every model, and every project that influenced the design, with its
license and whether any code was reused.

**Bottom line: no code from any GPL project was copied into this repository.** The two GPL
references (`axidex/face-retouching` and `georgegach/facestudio`) were studied for architecture
and algorithms only, and every algorithm here is implemented from published papers and public
documentation. Section 4 lists the specific influences.

**Disclaimer on accuracy.** The license statements below come from each project's declared
license, its README, its model card, or web search. They have not been reviewed by counsel.
Verify them again before shipping a commercial product, in particular the model weight licenses,
which are the highest-risk item.

---

## 1. Runtime dependencies

| package | version | license | purpose | code reused |
|---|---|---|---|---|
| [numpy](https://numpy.org) | >=1.24 | BSD-3-Clause | array math, the canonical buffer | no, used as a library |
| [opencv-contrib-python](https://github.com/opencv/opencv-python) | >=4.10 | Apache-2.0, **plus bundled LGPL binaries, see below** | box/Gaussian filters, color conversion, morphology, polygon fill, connected components, `remap` | no, used as a library |
| [mediapipe](https://github.com/google-ai-edge/mediapipe) | >=0.10.18 | Apache-2.0 | Face Landmarker inference | no, used as a library |
| [pillow](https://python-pillow.org) | >=10.0 | MIT-CMU (PIL license) | image IO with EXIF/ICC preservation and EXIF orientation | no, used as a library |

All four are permissive and compatible with a proprietary commercial product.

### Optional extras

| package | version | license | extra | purpose |
|---|---|---|---|---|
| [fastapi](https://fastapi.tiangolo.com) | >=0.110 | MIT | `api` | HTTP adapter |
| [uvicorn](https://www.uvicorn.org) | >=0.29 | BSD-3-Clause | `api` | ASGI server |
| [python-multipart](https://github.com/Kludex/python-multipart) | >=0.0.9 | Apache-2.0 | `api` | multipart upload parsing |
| [pytest](https://pytest.org) | >=8.0 | MIT | `dev` | tests |
| [ruff](https://docs.astral.sh/ruff/) | >=0.5 | MIT | `dev` | lint |
| [httpx2](https://pypi.org/project/httpx2/) | any | BSD-3-Clause | `dev` | required by Starlette's `TestClient` |

Deliberately **not** dependencies: PyTorch, PyWavelets, dlib, InsightFace, scikit-image, scipy,
and any CLI framework. Reasons in `docs/RESEARCH.md`.

### Correction, 2026-08-31: we ship `opencv-contrib-python`, and we cannot avoid it

This file previously said the in-repo guided filter "removes the need for `opencv-contrib-python`
entirely". That was wrong. **MediaPipe declares `opencv-contrib-python` as a hard dependency**
(`Requires-Dist: opencv-contrib-python` in its metadata), so it installs regardless of what we
ask for. Both packages install the same `cv2` module, so declaring `opencv-python` as well left
two providers of one import and made the resolved version depend on install order. The dependency
is now declared as `opencv-contrib-python`, leaving exactly one provider.

What this does and does not change:

- **It adds no binaries and no licence exposure.** Verified directly: `opencv_python.libs` and
  `opencv_contrib_python.libs` are byte-identical, same sha256 over their contents.
- **We still call nothing from the contrib modules.** Audited across the whole repository: every
  one of the ~40 distinct `cv2.*` symbols used is in the base build. The guided filter is still
  implemented in-repo rather than taken from `cv2.ximgproc`. That decision stands on its merits
  (it is calibrated against measurements this project recorded) but it is no longer *also* a way
  to avoid a dependency.

### The OpenCV wheel bundles LGPL components

`Apache-2.0` covers OpenCV's own source. The **wheel** additionally ships prebuilt binaries whose
licences are in `cv2/LICENSE-3RD-PARTY.txt`, 3617 lines, including the full texts of **LGPL 2.1
and LGPL 3** and a **Qt 5** notice; the bundled `libavcodec` / `libavformat` / `libavfilter` are
FFmpeg. Listing this dependency as flatly Apache-2.0 understated a live obligation.

For a closed-source commercial product this is workable but not free of duties: the LGPL parts
are dynamically linked shared objects, which is the permitted arrangement, and the obligations
are attribution plus preserving the user's ability to relink. **Do not vendor, statically link or
repackage these binaries without counsel.** If the obligation is ever unwanted, the escape is a
headless build of OpenCV without FFmpeg and Qt, which this engine would not miss: it decodes
images through Pillow and never opens a window.

---

## 2. Models

No model weights are vendored in this repository.

### MediaPipe Face Landmarker (used by default)

| | |
|---|---|
| file | `face_landmarker.task`, float16, v1 |
| source | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` |
| license | Apache-2.0, per Google's MediaPipe model card |
| obtained | downloaded on first use by `perception.landmarks.ensure_model()`, into `assets/models/` |
| override | `PORTRAIT_ENGINE_LANDMARKER_MODEL=/path/to/face_landmarker.task` |
| commercial use | believed permitted. **Not verified by counsel.** Confirm before shipping. |

The 478-point landmark index rings in `portrait_engine/perception/landmark_sets.py` were derived
once from `mediapipe/python/solutions/face_mesh_connections.py` (Apache-2.0, Google LLC) by
walking each connection set into an ordered closed ring, and frozen as literal tuples. This was
necessary because MediaPipe 1.x removed the `mediapipe.solutions` package, so the `FACEMESH_*`
constants can no longer be imported at runtime. **This is index data derived from Apache-2.0
licensed source**, which the Apache-2.0 license permits with attribution; that attribution is
this paragraph and the module docstring.

### BiSeNet face parsing: removed 2026-09-01

The optional BiSeNet backend and its `onnxruntime` extra have been **deleted from the project**.

It was off by default, shipped no weights, and was never exercised in any run or test. Its
reference implementation is MIT, but the weights everyone actually uses are trained on
CelebAMask-HQ, whose dataset licence is **non-commercial research only**, and a model trained on
non-commercially-licensed data carries that restriction in most readings. This was recorded here
as the single highest licensing risk in the project.

Keeping a dead module that is also a legal liability had no upside: the landmark-based masks are
the default, the engine works without it, and re-adding it is one `git revert`. The removal
deletes `perception/face_parsing.py`, the `use_parsing` and `parsing_model` options, the
`--use-parsing` flag, the `parsing_backend` field on `FaceAnalysis`, and the `parsing` extra.

**The project now has no non-commercial licensing exposure of any kind.**

If a commercially usable face-parsing model is needed later, the options are to train one on a
permissively licensed dataset, to license weights commercially, or to keep improving the
landmark-plus-skin-probability approach. The interface makes any of those a drop-in change.

---

## 3. Papers and documentation implemented from

Implementing from a paper produces original code and carries no license obligation from any
implementation of it.

| work | used for |
|---|---|
| He, Sun, Tang. *Guided Image Filtering*. ECCV 2010 / TPAMI 2013. | the guided filter in `core/ops.py`, the whole engine's smoothing and feathering primitive |
| He, Sun. *Fast Guided Filter*. arXiv 2015. | the `subsample` fast path |
| Velusamy, Parihar, Kini, Rege. *FabSoften: Face Beautification via Dynamic Skin Smoothing, Guided Feathering, and Texture Restoration*. CVPRW 2020. [IEEE](https://ieeexplore.ieee.org/document/9150825/) | the pipeline shape of `skin_smoothing`: approximate mask, skin probability, guided feathering, smooth then restore texture, attribute-driven dynamic parameters |
| Tomasi, Manduchi. *Bilateral Filtering for Gray and Color Images*. ICCV 1998. | evaluated and rejected, see `docs/RESEARCH.md` 4.1 |
| Telea. *An Image Inpainting Technique Based on the Fast Marching Method*. 2004. | evaluated and rejected for blemish healing, see `docs/RESEARCH.md` 4.4 |
| IEC 61966-2-1 (sRGB) | the exact transfer functions in `core/colorspace.py` |
| CIE 1976 L\*a\*b\* | the working perceptual space |

---

## 4. Projects studied, and exactly what was taken

### `axidex/face-retouching` — GPL-3.0

<https://github.com/axidex/face-retouching>

**Code reused: none. Influence: architecture and algorithm structure, from its public
documentation and from the FabSoften paper it implements.**

Taken as ideas:

1. The mask refinement sequence: approximate parsing mask, then a skin probability, then guided
   feathering into a soft alpha matte. Our `skin_probability` uses a per-image Gaussian in Lab
   rather than its GMM.
2. Smoothing parameters derived from measured attributes instead of constants. Ours scales the
   base radius by measured roughness; theirs scales an anisotropic diffusion radius and epsilon
   by skin probability and spot count.
3. The principle of smoothing and then restoring texture, rather than smoothing less. Ours does
   it by never attenuating the fine band; theirs by wavelet band manipulation.
4. The list of debug artifacts worth writing out.

Explicitly not taken: its DWT texture restoration (rejected, see `docs/RESEARCH.md` 4.3), its
GMM, its dlib dependency, and any line of its source.

### `georgegach/facestudio` — GPL-3.0-only

<https://github.com/georgegach/facestudio>

**Code reused: none. Influence: conceptual confirmation.**

Confirmed the pairing of MediaPipe Face Landmarker for dense landmarks with an optional
semantic parsing model, and the "perceive once, edit many times" split. Its WebGL2
piecewise-affine warp was deliberately not adopted: this engine forbids changing facial
geometry.

### OpenCV G-API face beautification tutorial — Apache-2.0

<https://docs.opencv.org/4.x/tutorials_contrib/gapi/face_beautification/face_beautification.html>

**Code reused: none (it is C++ G-API). Influence: the central design pattern, and two specific
techniques.**

This is the most important reference because it encodes the paradigm this project is built on:
deep learning for perception, classical computer vision for the edit, composited by masks.

Taken as ideas:

1. Composition through masks with weights that sum to one, so nothing outside a region can
   change. `core/masking.apply_masked` is a generalization of its `O = b*I + p*U + s*L`.
2. Blurring the mask rather than demanding pixel-accurate contours.
3. Interpolating closed contours from a handful of landmarks. It builds eye and forehead
   contours from 2 and 3 points respectively; we use the same idea for the infraorbital region,
   which has no landmark ring in the 478-point mesh.

Not taken: a global bilateral filter over the skin as the only smoothing step, which produces
the plastic result the brief prohibits, and its median-Laplacian unsharp mask, which amplifies
noise.

### `WWIIITT/PortraitRetouch` — no LICENSE file

<https://github.com/WWIIITT/PortraitRetouch>

**Code reused: none. Influence: backend structure only.**

No declared license means all rights reserved, so nothing was reused and nothing could be. Only
its publicly visible architecture was described: an agent planning layer separate from
deterministic processors, with FastAPI as a thin routing layer.

Taken as an idea: the separation of an agent from deterministic processors, and FastAPI as an
adapter rather than the core.

Deliberately improved on: it selects processors by endpoint and does not expose tool contracts
to the agent. This project exposes a generated JSON contract per tool
(`portrait_engine.tools.tool_schemas()`), which is what a tool-calling agent actually needs.

Rejected: its generative editing path via a Gemini-compatible API. This engine calls no external
API and uses no generative model.

---

## 5. Test fixtures

The evaluation portraits in `data/input/` come from Wikimedia Commons under CC BY 2.0,
CC BY-SA 3.0 and CC BY-SA 4.0. They are local fixtures only:

- `data/` is git-ignored except for `data/input/ATTRIBUTION.md`, which records file, source page,
  license and author;
- they are not redistributed with the package and are not referenced by any published artifact;
- `python scripts/fetch_test_images.py` regenerates them.

Because CC BY-SA is a share-alike license, these images must never be embedded in a distributed
build, in documentation that is published, or in a model training set.

---

## 6. Compatibility summary for a proprietary product

| item | safe for a closed commercial product |
|---|---|
| numpy, opencv-contrib-python, mediapipe, pillow | yes, with the LGPL attribution duty noted above |
| fastapi, uvicorn, python-multipart | yes (MIT / BSD / Apache-2.0) |
| MediaPipe Face Landmarker weights | believed yes (Apache-2.0 model card), **verify** |
| all code in `portrait_engine/` | yes, written for this project |
| FaceMesh index tuples | yes, derived from Apache-2.0 source, attributed |
| BiSeNet weights | n/a. The backend was removed on 2026-09-01; see above. |
| test fixtures in `data/` | not for redistribution (CC BY-SA share-alike) |
