"""Reproduce the `me_pro` professional result, now with the rewritten under-eye correction.

Not part of the engine. It is the exact recipe recorded in `docs/HANDOFF.md` section 8, kept as
a runnable script so the reference result can be regenerated after any change to the tools.

The one deliberate difference from the original `me_pro.jpg`: `reduce_dark_circles` runs at a
strength chosen against the *rewritten* tool. The original recipe used 0.30, but at the time the
tool delivered 2.6 percent of the measured deficit, so that number carried no meaning. See
`docs/ALGORITHMS.md` section 3.

    python scripts/render_me_pro.py me.jpeg --dark-circles 0.85 --ladder

Since `equalize_skin_tone` was promoted into the engine, the whole recipe is reachable through
the CLI and the API too. This script stays because it also renders the comparisons.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import cv2
import numpy as np

from portrait_engine import FaceAnalyzer, load_image, save_image
from portrait_engine.core.config import DEFAULT_CONFIG
from portrait_engine.debug import visualize as viz
from portrait_engine.diagnostics import summarize
from portrait_engine.tools.dark_circles import reduce_dark_circles
from portrait_engine.tools.face_lighting import balance_face_lighting
from portrait_engine.tools.skin_cleanup import skin_cleanup
from portrait_engine.tools.skin_smoothing import skin_smoothing
from portrait_engine.tools.skin_tone import equalize_skin_tone
from portrait_engine.tools.teeth_whitening import whiten_teeth

#: Wider and stronger on chroma than the engine default, gentler on the mid band. This is what
#: makes the skin read as "retouched by a person" on this photo: the dominant defect was regional
#: redness, not texture, so the work goes into chroma while the fine band is left at unity gain.
PRO_SKIN = replace(
    DEFAULT_CONFIG.skin_smoothing,
    max_mid_attenuation=0.75,
    max_chroma_attenuation=0.85,
    radius_chroma=0.24,
    eps_chroma=20.0,
    fine_gain=1.0,
)


def render(path: str, dark_circles: float) -> tuple[np.ndarray, dict]:
    """Run the recipe. Returns the finished RGB and the per-step stats."""
    image = load_image(path)
    analyzer = FaceAnalyzer()
    analysis = analyzer.analyze(image)
    rgb = image.rgb
    stats: dict = {"geometry": analysis.geometry.as_dict()}

    step = skin_cleanup(rgb, analysis, 0.85)
    rgb, stats["skin_cleanup"] = step.image, step.stats

    step = skin_smoothing(rgb, analysis, 0.55, config=PRO_SKIN)
    rgb, stats["skin_smoothing"] = step.image, step.stats

    step = equalize_skin_tone(rgb, analysis, 0.60)
    rgb, stats["equalize_skin_tone"] = step.image, step.stats

    step = reduce_dark_circles(rgb, analysis, dark_circles)
    rgb, stats["reduce_dark_circles"] = step.image, step.stats

    step = whiten_teeth(rgb, analysis, 0.20)
    rgb, stats["whiten_teeth"] = step.image, step.stats

    step = balance_face_lighting(rgb, analysis, 0.15)
    rgb, stats["balance_face_lighting"] = step.image, step.stats

    stats["diagnostics"] = summarize(
        image.rgb, rgb, analysis,
        ("skin_cleanup", "skin_smoothing", "equalize_skin_tone", "reduce_dark_circles",
         "whiten_teeth", "balance_face_lighting"),
        detector=analyzer.detector,
    )
    return rgb, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="me.jpeg")
    parser.add_argument("--dark-circles", type=float, default=0.85)
    parser.add_argument("--output", default="me_pro.jpg")
    parser.add_argument("--ladder", action="store_true",
                        help="also write an under-eye strength ladder at 100 percent")
    args = parser.parse_args()

    original = load_image(args.image)
    rgb, stats = render(args.image, args.dark_circles)

    # The source carries EXIF. For a personal photo that is worth stripping, even though the
    # engine preserves metadata by default, which is the right default for a real workflow.
    save_image(original.with_rgb(rgb), args.output, quality=96, keep_metadata=False)
    print(f"wrote {args.output}")

    analysis = FaceAnalyzer().analyze(original)
    comparison = viz.before_after(
        viz.crop_zoom(original.rgb, analysis, "face"),
        viz.crop_zoom(rgb, analysis, "face"),
        ("original", f"pro, dark circles {args.dark_circles:.2f}"),
    )
    name = args.output.replace(".jpg", "_comparison.jpg")
    cv2.imwrite(name, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 96])
    print(f"wrote {name}")

    if args.ladder:
        rows = []
        for value in (0.0, 0.4, 0.7, 1.0):
            variant, _ = render(args.image, value)
            strip = _eye_strip(variant, analysis)
            _label(strip, f"dark circles {value:.2f}")
            rows.append(strip)
        width = min(r.shape[1] for r in rows)
        stack = np.vstack([r[:, :width] for r in rows])
        cv2.imwrite("me_pro_undereye_ladder.jpg", cv2.cvtColor(stack, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 96])
        print("wrote me_pro_undereye_ladder.jpg")

    d = stats["diagnostics"]
    print(f"\ntexture retention   {d['texture_retention']}")
    print(f"out-of-region leak  {d['out_of_region_leak']}")
    print(f"landmark drift      {d['landmark_drift']}")
    print(f"changed fraction    {d['changed_pixel_fraction']}")
    print(f"warnings            {d['warnings']}")
    u = stats["reduce_dark_circles"]
    for side in ("left", "right"):
        if f"{side}_recovered_fraction" in u:
            print(f"under-eye {side}: deficit {u[f'{side}_mean_deficit_L']:.2f} L*, "
                  f"lift {u[f'{side}_mean_lift_L']:.2f}, "
                  f"residual {u[f'{side}_residual_deficit_L']:.2f}, "
                  f"recovered {u[f'{side}_recovered_fraction']:.3f}")


def _eye_strip(rgb: np.ndarray, analysis) -> np.ndarray:
    crops = [viz.crop_zoom(rgb, analysis, side) for side in ("right_eye", "left_eye")]
    height = min(c.shape[0] for c in crops)
    return viz._u8(np.hstack([c[:height] for c in crops]))


def _label(image: np.ndarray, text: str) -> None:
    for color, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(image, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness,
                    cv2.LINE_AA)


if __name__ == "__main__":
    main()
