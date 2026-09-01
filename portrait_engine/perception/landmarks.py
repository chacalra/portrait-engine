"""MediaPipe Face Landmarker backend.

This is the only model the engine runs by default. One inference gives the face box, 478
dense landmarks including irises, and a 4x4 head transformation matrix. Licensing and the
comparison against the alternatives are in ``docs/RESEARCH.md``.

The model file is not vendored. ``ensure_model`` downloads it once into ``assets/models/``
or uses ``PORTRAIT_ENGINE_LANDMARKER_MODEL`` if set.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from portrait_engine.core.types import BBox, NoFaceDetectedError, PortraitEngineError
from portrait_engine.perception import landmark_sets as ls

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "models",
)
MODEL_FILENAME = "face_landmarker.task"

_LOCK = threading.Lock()


@dataclass(frozen=True)
class DetectedFace:
    """One face as returned by the landmarker, already in pixel coordinates."""

    landmarks: np.ndarray  # (478, 2) float32 pixels
    landmarks_3d: np.ndarray  # (478, 3) float32, MediaPipe normalized space
    bbox: BBox
    transform: np.ndarray | None = None  # 4x4 head pose, informational only


def model_path() -> str:
    """Resolve the landmarker model path without downloading."""
    override = os.environ.get("PORTRAIT_ENGINE_LANDMARKER_MODEL")
    if override:
        return override
    return os.path.join(DEFAULT_MODEL_DIR, MODEL_FILENAME)


def ensure_model(path: str | None = None) -> str:
    """Return a local path to the landmarker model, downloading it once if needed."""
    target = path or model_path()
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        return target
    with _LOCK:
        if os.path.isfile(target) and os.path.getsize(target) > 0:
            return target
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        tmp = target + ".part"
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=120) as response, open(tmp, "wb") as out:
                out.write(response.read())
            os.replace(tmp, target)
        except Exception as exc:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise PortraitEngineError(
                f"could not obtain the Face Landmarker model ({exc}). Download it manually "
                f"from {MODEL_URL} into {target}."
            ) from exc
    return target


class LandmarkDetector:
    """Thin, thread-safe wrapper over ``vision.FaceLandmarker`` in IMAGE mode.

    Instances are expensive (they build a TFLite graph), so reuse one. ``shared()`` returns a
    process-wide singleton, which is what ``FaceAnalyzer`` uses.
    """

    def __init__(self, model: str | None = None, max_faces: int = 5, min_confidence: float = 0.5):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover
            raise PortraitEngineError(
                "mediapipe is required for the default perception backend: pip install mediapipe"
            ) from exc

        self._mp = mp
        self._model = ensure_model(model)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self._model),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model

    def detect(self, rgb: np.ndarray) -> list[DetectedFace]:
        """Detect faces in an sRGB ``float32 [0,1]`` or ``uint8`` RGB image.

        MediaPipe wants ``uint8`` RGB, so this is the one place the canonical float buffer is
        quantized. That only affects detection, never the pixels that get written out.
        """
        if rgb.dtype != np.uint8:
            frame = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
        else:
            frame = rgb
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)
        with self._lock:  # the underlying graph is not reentrant
            result = self._landmarker.detect(mp_image)

        faces: list[DetectedFace] = []
        matrices = getattr(result, "facial_transformation_matrixes", None) or []
        for index, points in enumerate(result.face_landmarks):
            xyz = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
            pixels = np.stack([xyz[:, 0] * width, xyz[:, 1] * height], axis=1).astype(np.float32)
            faces.append(
                DetectedFace(
                    landmarks=pixels,
                    landmarks_3d=xyz,
                    bbox=bbox_from_landmarks(pixels, (height, width)),
                    transform=np.asarray(matrices[index], dtype=np.float32)
                    if index < len(matrices)
                    else None,
                )
            )
        return faces

    def detect_one(self, rgb: np.ndarray) -> DetectedFace:
        faces = self.detect(rgb)
        if not faces:
            raise NoFaceDetectedError(
                "no face detected. The MVP expects one clearly visible, frontal or "
                "semi-frontal face occupying a reasonable part of the frame."
            )
        return max(faces, key=lambda f: f.bbox.width * f.bbox.height)

    @staticmethod
    @lru_cache(maxsize=4)
    def shared(model: str | None = None, max_faces: int = 5) -> LandmarkDetector:
        return LandmarkDetector(model=model, max_faces=max_faces)


def bbox_from_landmarks(landmarks: np.ndarray, shape: tuple[int, int]) -> BBox:
    """Tight box around the face oval landmarks, clipped to the frame."""
    oval = landmarks[list(ls.FACE_OVAL)]
    x0, y0 = np.floor(oval.min(axis=0)).astype(int)
    x1, y1 = np.ceil(oval.max(axis=0)).astype(int)
    return BBox(int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))).clipped(shape)
