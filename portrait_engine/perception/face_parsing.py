"""Optional BiSeNet face-parsing backend. Disabled by default.

Why optional
------------
The parsing model would improve exactly one thing the landmark-based masks are weak at:
separating skin from hair, beard and neck. It is not enabled by default because of a
licensing risk that matters for a commercial product: the widely used weights
(``zllrunning/face-parsing.PyTorch`` and its ONNX exports) are MIT-licensed *code* trained on
**CelebAMask-HQ, whose dataset license is non-commercial**. Details in
``docs/THIRD_PARTY.md``.

No weights are vendored. Point ``PORTRAIT_ENGINE_PARSING_MODEL`` at a local ONNX file, or
pass ``model_path`` explicitly, and accept the licensing consequences yourself.

Contract
--------
A parser maps an RGB image to per-class alpha masks using this engine's mask names, so the
rest of the code does not care which backend produced them.
"""

from __future__ import annotations

import os
from typing import Protocol

import cv2
import numpy as np

from portrait_engine.core.types import PortraitEngineError

#: The 19 CelebAMask-HQ classes in the order BiSeNet outputs them.
BISENET_CLASSES: tuple[str, ...] = (
    "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "l_ear", "r_ear",
    "ear_r", "nose", "mouth", "u_lip", "l_lip", "neck", "neck_l", "cloth", "hair", "hat",
)

#: How BiSeNet classes map onto this engine's mask names.
CLASS_TO_MASK: dict[str, tuple[str, ...]] = {
    "skin": ("parsed_skin",),
    "hair": ("parsed_hair",),
    "hat": ("parsed_hair",),
    "l_brow": ("parsed_eyebrows",),
    "r_brow": ("parsed_eyebrows",),
    "l_eye": ("parsed_eyes",),
    "r_eye": ("parsed_eyes",),
    "eye_g": ("parsed_glasses",),
    "u_lip": ("parsed_lips",),
    "l_lip": ("parsed_lips",),
    "mouth": ("parsed_mouth_interior",),
    "nose": ("parsed_nose",),
    "neck": ("parsed_neck",),
    "cloth": ("parsed_cloth",),
}


class FaceParser(Protocol):
    """Anything that can turn an RGB image into named alpha masks."""

    def parse(self, rgb: np.ndarray) -> dict[str, np.ndarray]: ...


def model_path() -> str | None:
    return os.environ.get("PORTRAIT_ENGINE_PARSING_MODEL")


def is_available(path: str | None = None) -> bool:
    target = path or model_path()
    if not target or not os.path.isfile(target):
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


class BiSeNetParser:
    """ONNX Runtime wrapper around a BiSeNet face-parsing model.

    Expected model signature: input ``(1, 3, 512, 512)`` float32, ImageNet-normalized RGB;
    output ``(1, 19, 512, 512)`` logits. That is what the standard exports produce.
    """

    INPUT_SIZE = 512
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path_: str | None = None, providers: list[str] | None = None):
        target = model_path_ or model_path()
        if not target or not os.path.isfile(target):
            raise PortraitEngineError(
                "no face-parsing model found. Set PORTRAIT_ENGINE_PARSING_MODEL to a local "
                "BiSeNet ONNX file. Review its weight license before using it commercially "
                "(see docs/THIRD_PARTY.md)."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise PortraitEngineError("onnxruntime is required for face parsing") from exc

        self._session = ort.InferenceSession(
            target, providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.model_path = target

    def parse(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        height, width = rgb.shape[:2]
        resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_AREA)
        normalized = (resized - self.MEAN) / self.STD
        tensor = np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)

        logits = self._session.run(None, {self._input_name: tensor})[0][0]
        labels = np.argmax(logits, axis=0).astype(np.uint8)
        labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)

        masks: dict[str, np.ndarray] = {}
        for index, class_name in enumerate(BISENET_CLASSES):
            for mask_name in CLASS_TO_MASK.get(class_name, ()):
                layer = (labels == index).astype(np.float32)
                masks[mask_name] = np.maximum(masks.get(mask_name, 0.0), layer)
        return masks
