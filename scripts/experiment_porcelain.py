"""EXPERIMENT, not part of the engine: how far can this engine push a porcelain skin look?

Nothing here is imported by `portrait_engine`. It exists so the finding below is reproducible
and so the decision about whether to productize it can be made on evidence.

    python scripts/experiment_porcelain.py [image ...] [--out data/output/porcelain]

Finding
-------
A porcelain look needs four ingredients. Three of them are just the existing skin smoothing
config pushed past its safe ceilings, and one is a component the engine does not have:

1. `max_mid_attenuation` 0.85 -> 1.00     (already a config field)
2. `max_chroma_attenuation` 0.70 -> 0.95  (already a config field)
3. `radius_base` 0.130 -> 0.200           (already a config field)
4. `fine_gain` 1.00 -> ~0.35              (already a config field, but it is what the engine
                                           deliberately never touches: this is the pore band)
5. **bloom**, a new component: an Orton-style lift toward white in linear light, weighted by a
   blurred luma. This is what supplies the luminous quality. Without it the result reads as
   "smoothed", not "porcelain".

A fifth, optional component, `crease_soften`, attenuates the elongated dark structures that the
mid band leaves behind (crow's feet, nasolabial folds, forehead lines).

**`crease_soften` is what makes the look work on a face that has structural wrinkles, and it is
also what the brief forbids**: requirement 7, "no cambiar edad aparente", and the exclusion list
item "eliminación de arrugas estructurales". Measured on a middle-aged subject it takes roughly
10 to 15 years off the apparent age. It is therefore off by default here and printed with a
warning when enabled.

Without it, the porcelain look is convincing on subjects who already have little structural
wrinkling and uncanny on those who do not: smooth skin with intact crow's feet.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

import cv2
import numpy as np

from portrait_engine import FaceAnalyzer, load_image, skin_smoothing
from portrait_engine.core import ops
from portrait_engine.core.colorspace import (
    LUMA_709,
    lab_to_rgb,
    linear_to_srgb,
    rgb_to_lab,
    soft_highlight_rolloff,
    srgb_to_linear,
)
from portrait_engine.core.config import DEFAULT_CONFIG
from portrait_engine.core.masking import apply_masked
from portrait_engine.debug import visualize as viz

#: Skin smoothing pushed past every safe ceiling in `SkinSmoothingConfig`.
PORCELAIN_SMOOTHING = replace(
    DEFAULT_CONFIG.skin_smoothing,
    max_mid_attenuation=1.00,
    max_chroma_attenuation=0.95,
    radius_base=0.200,
    fine_gain=0.35,
)


def bloom(rgb, mask, analysis, strength: float, radius_fraction: float = 0.10):
    """Orton-style bloom in linear light: lift toward white, weighted by a blurred luminance.

    In linear light so it behaves like light spilling, not like a curve. The lift is scaled by
    `(1 - linear)` so it fills shadows and pores and leaves highlights alone, and it ends in a
    soft rolloff so nothing clips.
    """
    linear = srgb_to_linear(rgb)
    luma = (linear @ LUMA_709).astype(np.float32)
    radius = analysis.geometry.scaled(radius_fraction, 3)
    kernel = 2 * radius + 1
    blurred = cv2.GaussianBlur(luma, (kernel, kernel), 0, borderType=cv2.BORDER_REFLECT101)
    lifted = linear + strength * blurred[..., None] * (1.0 - linear)
    edited = linear_to_srgb(soft_highlight_rolloff(lifted, knee=0.90))
    return apply_masked(rgb, edited, mask, strict=True, name="bloom")


def crease_soften(rgb, mask, analysis, strength: float):
    """Attenuate elongated dark creases. VIOLATES the brief's identity rules, see module docstring.

    Only pixels darker than an edge-preserving smooth of themselves are lifted, so this targets
    creases rather than highlights. That is also exactly why it removes wrinkles.
    """
    lab = rgb_to_lab(rgb)
    lightness = lab[..., 0]
    radius = analysis.geometry.scaled(0.045, 2)
    smooth = ops.guided_filter(lightness, lightness, radius, 250.0, subsample=2)
    new_l = lightness + strength * np.maximum(smooth - lightness, 0.0)
    edited = lab_to_rgb(np.stack([new_l, lab[..., 1], lab[..., 2]], axis=-1))
    return apply_masked(rgb, edited, mask, strict=True, name="crease_soften")


def porcelain(rgb, analysis, *, bloom_strength=0.16, crease_strength=0.0):
    skin = analysis.masks["skin"]
    out = skin_smoothing(rgb, analysis, 1.0, config=PORCELAIN_SMOOTHING).image
    if crease_strength > 0.0:
        out = crease_soften(out, skin, analysis, crease_strength)
    if bloom_strength > 0.0:
        out = bloom(out, skin, analysis, bloom_strength)
    return out


def measure(before, after, analysis, analyzer):
    from portrait_engine import diagnostics

    skin = analysis.masks["skin"]
    before_l = rgb_to_lab(before)[..., 0]
    after_l = rgb_to_lab(after)[..., 0]
    leak = diagnostics.out_of_region_leak(before, after, skin)
    drift = diagnostics.landmark_drift(before, after, analysis, analyzer.detector)
    return {
        "pore_retention": ops.high_frequency_energy(after_l, skin)
        / max(ops.high_frequency_energy(before_l, skin), 1e-6),
        "leak_max_levels": leak["max_levels"],
        "landmark_drift_rms": drift["rms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", default=None)
    parser.add_argument("--out", default="data/output/porcelain")
    parser.add_argument("--bloom", type=float, default=0.16)
    parser.add_argument(
        "--crease", type=float, default=0.0,
        help="attenuate wrinkles. Changes apparent age, which the brief forbids. Off by default.",
    )
    args = parser.parse_args()

    images = args.images or sorted(
        os.path.join("data/input", f) for f in os.listdir("data/input") if f.endswith(".jpg")
    )
    os.makedirs(args.out, exist_ok=True)
    if args.crease > 0:
        print(
            "WARNING: --crease attenuates structural wrinkles and measurably lowers apparent age.\n"
            "         Requirement 7 of the brief forbids this. Use only deliberately.\n"
        )

    analyzer = FaceAnalyzer()
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            image = load_image(path, max_side=1600)
            analysis = analyzer.analyze(image)
        except Exception as exc:
            print(f"  skip  {name}: {exc}")
            continue

        result = porcelain(image.rgb, analysis, bloom_strength=args.bloom, crease_strength=args.crease)
        stats = measure(image.rgb, result, analysis, analyzer)
        print(
            f"  {name[:40]:41s} pore={stats['pore_retention']:.3f} "
            f"leak={stats['leak_max_levels']:.3f}/255 drift={stats['landmark_drift_rms']:.5f}"
        )

        cv2.imwrite(
            os.path.join(args.out, f"{name}.jpg"),
            cv2.cvtColor(
                viz.before_after(
                    viz.crop_zoom(viz._u8(image.rgb), analysis, "face") / 255.0,
                    viz.crop_zoom(viz._u8(result), analysis, "face") / 255.0,
                    ("original", "porcelain"),
                ),
                cv2.COLOR_RGB2BGR,
            ),
        )


if __name__ == "__main__":
    main()
