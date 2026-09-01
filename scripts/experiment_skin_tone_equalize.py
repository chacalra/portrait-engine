"""EXPERIMENT, not part of the engine: equalizing skin tone between REGIONS of a face.

Nothing here is imported by `portrait_engine`.

The gap this exposes
--------------------
`skin_smoothing` removes *local* unevenness: it pulls every pixel toward its own local mean.
That fixes blotches. It does nothing at all for **regional** ruddiness, because when a whole
cheek is uniformly red the local mean is red too, so there is nothing to pull toward.

Measured on a real portrait with ruddy cheeks (a* by region: forehead 14.2, temple 13.4, nose
18.4, cheeks 19.9 and 19.3, a spread of 6.5):

| approach | region range |
|---|---|
| original | 6.5 |
| `skin_smoothing`, `radius_chroma` 0.14, attenuation 0.70 (default) | 5.8 |
| `skin_smoothing`, `radius_chroma` 0.45, attenuation 0.90 | 4.5 |
| `equalize_skin_tone` at 0.9 | **3.2** |

Widening the chroma radius barely helps, and it costs a lot of radius to get there. The
operation below reaches further with a much smaller change, because it is the right operation
for the defect: estimate the very-low-frequency chroma field over the skin and shift it toward
one face-wide target. Local detail is untouched by construction; measured pore energy went
4.91 -> 4.91, unchanged to three significant figures at every strength tested.

This is what a retoucher does with a hue/saturation adjustment or a color dodge-and-burn pass,
and it is a genuine missing capability rather than a tuning problem.

    python scripts/experiment_skin_tone_equalize.py me.jpeg
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from portrait_engine import FaceAnalyzer, load_image
from portrait_engine.core import ops
from portrait_engine.core.colorspace import lab_to_rgb, rgb_to_lab
from portrait_engine.core.masking import apply_masked
from portrait_engine.debug import visualize as viz


def equalize_skin_tone(rgb, mask, analysis, strength: float, radius_fraction: float = 0.30):
    """Pull each region of the skin's chroma toward the face's own median.

    Args:
        strength: 0 leaves the image untouched. 1.0 would move each region's low-frequency
            chroma all the way onto the face median, which is too far: some regional variation
            (a warmer cheek, a cooler temple) is what makes skin look like skin. 0.5 to 0.7 is
            the useful range.
        radius_fraction: size of the chroma field estimate, as a fraction of the interocular
            distance. Must be large: it has to average over a whole region, not a blotch.

    Only ``a*`` and ``b*`` are touched. ``L*`` is not, so modelling and texture are unaffected.
    """
    lab = rgb_to_lab(rgb)
    radius = analysis.geometry.scaled(radius_fraction, 8)
    channels = [lab[..., 0]]
    for index in (1, 2):
        field = ops.normalized_lowpass(lab[..., index], mask, radius)
        target = ops.masked_percentile(lab[..., index], mask, 50.0)
        if not np.isfinite(target):
            channels.append(lab[..., index])
            continue
        channels.append(lab[..., index] + strength * (target - field))
    edited = lab_to_rgb(np.stack(channels, axis=-1))
    return apply_masked(rgb, edited, mask, strict=True, name="equalize_skin_tone")


def region_range(rgb, analysis, mask) -> float:
    """Spread of median a* across forehead, nose, both cheeks and a temple. The defect, as one
    number."""
    from portrait_engine.perception.masks import _disc

    a_star = rgb_to_lab(rgb)[..., 1]
    selection = mask > 0.5
    values = []
    for index in (151, 1, 205, 425, 234):
        disc = (_disc(rgb.shape[:2], analysis.landmarks[index], analysis.geometry.scaled(0.09, 4)) > 0.5) & selection
        if disc.sum() > 50:
            values.append(float(np.median(a_star[disc])))
    return float(max(values) - min(values)) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--out", default="data/output/skin_tone")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    image = load_image(args.image)
    analysis = FaceAnalyzer().analyze(image)
    skin = analysis.masks["skin"]

    tiles = [viz.crop_zoom(viz._u8(image.rgb), analysis, "face")]
    labels = ["original"]
    print(f"original           region a* range = {region_range(image.rgb, analysis, skin):.1f}")
    for strength in (0.3, 0.6, 0.9):
        result = equalize_skin_tone(image.rgb, skin, analysis, strength)
        pore = ops.high_frequency_energy(rgb_to_lab(result)[..., 0], skin)
        print(
            f"equalize {strength:.1f}       region a* range = "
            f"{region_range(result, analysis, skin):.1f}   pore energy = {pore:.2f}"
        )
        tiles.append(viz.crop_zoom(viz._u8(result), analysis, "face"))
        labels.append(f"equalize {strength:.1f}")

    height = min(t.shape[0] for t in tiles)
    width = min(t.shape[1] for t in tiles)
    strip = np.hstack([t[:height, :width] for t in tiles])
    for index, label in enumerate(labels):
        for color, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(strip, label, (index * width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness, cv2.LINE_AA)
    path = os.path.join(args.out, "equalize_sweep.jpg")
    cv2.imwrite(path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"\n{path}")


if __name__ == "__main__":
    main()
