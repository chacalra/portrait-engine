# Models

Model files are **not** vendored in this repository. They are downloaded on first use.

| file | source | license | downloaded by |
|---|---|---|---|
| `face_landmarker.task` | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (float16, v1) | Apache-2.0 | `portrait_engine.perception.landmarks.ensure_model()` on first run |

Override the location with `PORTRAIT_ENGINE_LANDMARKER_MODEL=/path/to/face_landmarker.task`.

## Optional face parsing

The BiSeNet backend is off by default and no weights are provided. Point
`PORTRAIT_ENGINE_PARSING_MODEL` at a local ONNX file if you want it, and read
`docs/THIRD_PARTY.md` first: the commonly used weights are trained on CelebAMask-HQ, whose
dataset license is non-commercial.
