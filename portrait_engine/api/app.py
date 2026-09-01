"""Minimal FastAPI adapter.

Deliberately thin. It parses requests, calls the engine, and serializes responses. There is no
business logic here, no state, and nothing the library cannot do on its own. If this file were
deleted the engine would be unaffected, which is the property we wanted.

Run it with::

    pip install "portrait-engine[api]"
    uvicorn portrait_engine.api.app:app --reload

Endpoints:

* ``POST /analyze``  multipart image -> face box, geometry, mask areas.
* ``POST /retouch``  multipart image plus strengths -> retouched JPEG.
* ``GET  /tools``    the JSON tool contracts, for a future agent.
* ``GET  /health``
"""

from __future__ import annotations

import time
from typing import Annotated, Any

from portrait_engine import __version__
from portrait_engine.core.image import PortraitImage, decode_bytes, encode_jpeg, resize_long_side
from portrait_engine.core.pipeline import PortraitEngine
from portrait_engine.core.types import (
    InvalidImageError,
    MultipleFacesError,
    NoFaceDetectedError,
    PortraitEngineError,
    StrengthOutOfRangeError,
)
from portrait_engine.tools import tool_schemas

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, Response
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'the API extra is required: pip install "portrait-engine[api]"'
    ) from exc

#: Upload ceiling. Portraits, not panoramas.
MAX_UPLOAD_BYTES = 40 * 1024 * 1024

#: Long-side cap applied to every request, so one large upload cannot monopolize the process.
MAX_SIDE = 2400

app = FastAPI(
    title="Portrait Retouching Engine",
    version=__version__,
    summary="Deterministic portrait retouching. Perception by MediaPipe, edits by classical "
            "image processing. No generative models, no facial geometry changes.",
)

_engine: PortraitEngine | None = None


def get_engine() -> PortraitEngine:
    """Lazily build one shared engine. Model construction is expensive; requests are not."""
    global _engine
    if _engine is None:
        _engine = PortraitEngine()
    return _engine


async def _read_image(upload: UploadFile) -> PortraitImage:
    payload = await upload.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"upload is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        image = decode_bytes(payload)
    except InvalidImageError as exc:
        raise HTTPException(400, str(exc)) from exc
    return resize_long_side(image, MAX_SIDE)


def _map_error(exc: PortraitEngineError) -> HTTPException:
    """Perception outcomes are client errors, not server errors."""
    if isinstance(exc, NoFaceDetectedError):
        return HTTPException(422, {"error": "no_face_detected", "detail": str(exc)})
    if isinstance(exc, MultipleFacesError):
        return HTTPException(422, {"error": "multiple_faces", "detail": str(exc)})
    if isinstance(exc, StrengthOutOfRangeError):
        return HTTPException(400, {"error": "invalid_strength", "detail": str(exc)})
    if isinstance(exc, InvalidImageError):
        return HTTPException(400, {"error": "invalid_image", "detail": str(exc)})
    return HTTPException(500, {"error": "engine_error", "detail": str(exc)})


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__}


@app.get("/tools")
def tools() -> list[dict[str, Any]]:
    """The tool contracts, ready to hand to an LLM tool-calling loop."""
    return tool_schemas()


@app.post("/analyze")
async def analyze(file: Annotated[UploadFile, File(description="JPG or PNG portrait")]) -> JSONResponse:
    """Run perception only and return what was detected."""
    image = await _read_image(file)
    started = time.perf_counter()
    try:
        _, analysis = get_engine().analyze(image)
    except PortraitEngineError as exc:
        raise _map_error(exc) from exc
    payload = analysis.summary()
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return JSONResponse(payload)


@app.post("/retouch")
async def retouch(
    file: Annotated[UploadFile, File(description="JPG or PNG portrait")],
    skin_cleanup: Annotated[float, Form(ge=0.0, le=1.0)] = 0.0,
    skin_smoothing: Annotated[float, Form(ge=0.0, le=1.0)] = 0.0,
    dark_circles: Annotated[float, Form(ge=0.0, le=1.0)] = 0.0,
    teeth_whitening: Annotated[float, Form(ge=0.0, le=1.0)] = 0.0,
    face_lighting: Annotated[float, Form(ge=0.0, le=1.0)] = 0.0,
    quality: Annotated[int, Form(ge=60, le=100)] = 95,
    diagnostics: Annotated[bool, Form()] = True,
) -> Response:
    """Retouch and return a JPEG.

    The report is returned in the ``X-Portrait-Report`` header as JSON, so the body stays a
    plain image that a browser or an ``<img>`` tag can use directly.
    """
    image = await _read_image(file)
    strengths = {
        "skin_cleanup": skin_cleanup,
        "skin_smoothing": skin_smoothing,
        "dark_circles": dark_circles,
        "teeth_whitening": teeth_whitening,
        "face_lighting": face_lighting,
    }
    if not any(value > 0 for value in strengths.values()):
        raise HTTPException(400, {"error": "no_operation", "detail": "every strength is zero"})

    started = time.perf_counter()
    engine = get_engine()
    try:
        result = engine.process(image, run_diagnostics=diagnostics, **strengths)
    except PortraitEngineError as exc:
        raise _map_error(exc) from exc

    import json

    report = result.report()
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    body = encode_jpeg(PortraitImage(rgb=result.image, meta=image.meta), quality=quality)
    return Response(
        content=body,
        media_type="image/jpeg",
        headers={
            "X-Portrait-Report": json.dumps(report, default=str),
            "Content-Disposition": 'inline; filename="retouched.jpg"',
        },
    )
