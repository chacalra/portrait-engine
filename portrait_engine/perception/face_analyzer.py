"""``FaceAnalyzer``: run perception once, hand every tool the same ``FaceAnalysis``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portrait_engine.core.image import PortraitImage
from portrait_engine.core.types import (
    FaceAnalysis,
    MultipleFacesError,
    NoFaceDetectedError,
)
from portrait_engine.perception.landmarks import LandmarkDetector
from portrait_engine.perception.masks import (
    MaskBuildConfig,
    build_masks,
    geometry_from_landmarks,
)

#: A face smaller than this fraction of the frame's long side gives unreliable masks.
MIN_FACE_FRACTION = 0.06


@dataclass
class AnalyzerConfig:
    """Perception policy.

    ``allow_multiple_faces`` stays False in the MVP by design: every tool assumes one subject,
    and silently retouching only the largest face would be a worse failure than an error.
    """

    max_faces: int = 5
    allow_multiple_faces: bool = False
    min_detection_confidence: float = 0.5
    mask_config: MaskBuildConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mask_config is None:
            self.mask_config = MaskBuildConfig()


class FaceAnalyzer:
    """Detects the face and builds every mask the tools need.

    One instance can be reused across images and is safe to share between threads. The
    expensive parts (the TFLite graph, the ONNX session) are created once.
    """

    def __init__(self, config: AnalyzerConfig | None = None):
        self.config = config or AnalyzerConfig()
        self._detector = LandmarkDetector.shared(None, self.config.max_faces)

    @property
    def detector(self) -> LandmarkDetector:
        return self._detector

    def analyze(self, image: PortraitImage | np.ndarray) -> FaceAnalysis:
        """Run perception on one image and return an immutable ``FaceAnalysis``.

        Raises:
            NoFaceDetectedError: nothing detected, or the face is too small to work on.
            MultipleFacesError: more than one face and ``allow_multiple_faces`` is False.
        """
        rgb = image.rgb if isinstance(image, PortraitImage) else np.asarray(image, dtype=np.float32)
        shape = (rgb.shape[0], rgb.shape[1])

        faces = self._detector.detect(rgb)
        if not faces:
            raise NoFaceDetectedError(
                "no face detected. The MVP expects one clearly visible, frontal or "
                "semi-frontal face. Try a tighter crop of the subject."
            )
        if len(faces) > 1 and not self.config.allow_multiple_faces:
            raise MultipleFacesError(
                f"{len(faces)} faces detected. The MVP handles exactly one person; "
                "crop to a single subject."
            )
        face = max(faces, key=lambda f: f.bbox.width * f.bbox.height)

        long_side = max(shape)
        if max(face.bbox.width, face.bbox.height) < MIN_FACE_FRACTION * long_side:
            raise NoFaceDetectedError(
                f"the detected face is only {max(face.bbox.width, face.bbox.height)} px on a "
                f"{long_side} px frame. Masks would be unreliable below "
                f"{MIN_FACE_FRACTION:.0%} of the long side. Crop closer."
            )

        geometry = geometry_from_landmarks(face.landmarks)
        masks, skin_prob, stats = build_masks(rgb, face, geometry, self.config.mask_config)

        analysis = FaceAnalysis(
            image_shape=shape,
            bbox=face.bbox,
            landmarks=face.landmarks,
            landmarks_3d=face.landmarks_3d,
            masks=masks,
            skin_probability=skin_prob,
            geometry=geometry,
        )
        object.__setattr__(analysis, "_stats", stats)  # kept for debug reporting
        return analysis
