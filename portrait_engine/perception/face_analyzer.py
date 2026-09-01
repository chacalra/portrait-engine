"""``FaceAnalyzer``: run perception once, hand every tool the same ``FaceAnalysis``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portrait_engine.core import ops
from portrait_engine.core.image import PortraitImage
from portrait_engine.core.types import (
    FaceAnalysis,
    MultipleFacesError,
    NoFaceDetectedError,
)
from portrait_engine.perception import face_parsing
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
    use_parsing: bool = False
    parsing_model: str | None = None
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
        self._parser: face_parsing.FaceParser | None = None
        if self.config.use_parsing:
            self._parser = face_parsing.BiSeNetParser(self.config.parsing_model)

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
        backend = "landmarks"

        if self._parser is not None:
            masks, skin_prob = self._refine_with_parsing(rgb, masks, skin_prob, geometry)
            backend = "landmarks+bisenet"

        analysis = FaceAnalysis(
            image_shape=shape,
            bbox=face.bbox,
            landmarks=face.landmarks,
            landmarks_3d=face.landmarks_3d,
            masks=masks,
            skin_probability=skin_prob,
            geometry=geometry,
            parsing_backend=backend,
        )
        object.__setattr__(analysis, "_stats", stats)  # kept for debug reporting
        return analysis

    def _refine_with_parsing(self, rgb, masks, skin_prob, geometry):
        """Intersect the landmark skin mask with the parsed skin class, and subtract hair.

        Parsing is used to *narrow* masks, never to widen them: a segmentation error should
        not hand a tool permission to edit somewhere new.
        """
        parsed = self._parser.parse(rgb)  # type: ignore[union-attr]
        feather = geometry.scaled(0.05, 2)
        extra: dict[str, np.ndarray] = dict(parsed)

        parsed_skin = parsed.get("parsed_skin")
        parsed_hair = parsed.get("parsed_hair")
        if parsed_skin is not None:
            soft_skin = ops.feather_mask(parsed_skin, rgb, feather, eps=1e-3, subsample=2)
            new_skin = np.clip(masks["skin"] * np.maximum(soft_skin, 0.25), 0.0, 1.0)
            if parsed_hair is not None:
                soft_hair = ops.feather_mask(parsed_hair, rgb, feather, eps=1e-3, subsample=2)
                new_skin = np.clip(new_skin * (1.0 - soft_hair), 0.0, 1.0)
                extra["hair_like"] = np.clip(
                    np.maximum(masks["hair_like"], soft_hair), 0.0, 1.0
                )
                extra["protected"] = np.clip(
                    np.maximum(masks["protected"], soft_hair), 0.0, 1.0
                )
            extra["skin"] = new_skin
            skin_prob = np.clip(skin_prob * np.maximum(soft_skin, 0.2), 0.0, 1.0)

        return masks.with_masks(extra), skin_prob
