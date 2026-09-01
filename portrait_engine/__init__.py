"""Portrait Retouching Engine.

A small, programmable image-processing SDK for professional portrait retouching. Deep learning
is used only to *perceive* the face; every edit is a deterministic operation on the original
pixels. No generative model is involved, and facial geometry is never modified.

Quick start::

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

Tools can also be used one at a time::

    from portrait_engine import FaceAnalyzer, load_image, whiten_teeth

    image = load_image("portrait.jpg")
    analysis = FaceAnalyzer().analyze(image)
    result = whiten_teeth(image, analysis, strength=0.15)
"""

from portrait_engine.core.config import DEFAULT_CONFIG, EngineConfig
from portrait_engine.core.image import (
    ImageMeta,
    PortraitImage,
    decode_bytes,
    encode_jpeg,
    from_array,
    load_image,
    save_image,
)
from portrait_engine.core.pipeline import PortraitEngine
from portrait_engine.core.types import (
    BBox,
    FaceAnalysis,
    FaceGeometry,
    InvalidImageError,
    MaskContractError,
    MaskSet,
    MultipleFacesError,
    NoFaceDetectedError,
    PortraitEngineError,
    RetouchResult,
    StrengthOutOfRangeError,
    ToolResult,
)
from portrait_engine.perception.face_analyzer import AnalyzerConfig, FaceAnalyzer
from portrait_engine.tools import (
    TOOL_REGISTRY,
    balance_face_lighting,
    reduce_dark_circles,
    skin_cleanup,
    skin_smoothing,
    tool_schemas,
    whiten_teeth,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_CONFIG",
    "TOOL_REGISTRY",
    "AnalyzerConfig",
    "BBox",
    "EngineConfig",
    "FaceAnalysis",
    "FaceAnalyzer",
    "FaceGeometry",
    "ImageMeta",
    "InvalidImageError",
    "MaskContractError",
    "MaskSet",
    "MultipleFacesError",
    "NoFaceDetectedError",
    "PortraitEngine",
    "PortraitEngineError",
    "PortraitImage",
    "RetouchResult",
    "StrengthOutOfRangeError",
    "ToolResult",
    "__version__",
    "balance_face_lighting",
    "decode_bytes",
    "encode_jpeg",
    "from_array",
    "load_image",
    "reduce_dark_circles",
    "save_image",
    "skin_cleanup",
    "skin_smoothing",
    "tool_schemas",
    "whiten_teeth",
]
