"""Run one tool on its own, and read what it measured.

    python examples/01_single_tool.py data/input/portrait.jpg
"""

from __future__ import annotations

import json
import sys

from portrait_engine import FaceAnalyzer, load_image, save_image, whiten_teeth


def main(path: str) -> None:
    image = load_image(path, max_side=1600)

    # Perception runs once and is reusable across as many tools as you like.
    analysis = FaceAnalyzer().analyze(image)
    print("face:", json.dumps(analysis.summary()["geometry"], indent=2))
    print("teeth mask area:", round(analysis.masks.area("teeth"), 1), "px")

    result = whiten_teeth(image, analysis, strength=0.15)
    if not result.changed:
        print("no-op:", result.stats["reason"])
        return

    print("stats:", json.dumps(result.stats, indent=2, default=str))
    save_image(image.with_rgb(result.image), "output/example_teeth.jpg")
    print("wrote output/example_teeth.jpg")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/input/laughing_woman_with_teeth.jpg")
