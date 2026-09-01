"""The full pipeline with debug output, and what to look at in it.

    python examples/02_full_pipeline.py data/input/portrait.jpg
"""

from __future__ import annotations

import json
import sys

from portrait_engine import PortraitEngine


def main(path: str) -> None:
    engine = PortraitEngine()
    result = engine.process(
        path,
        max_side=1600,
        skin_cleanup=0.15,
        skin_smoothing=0.20,
        dark_circles=0.15,
        teeth_whitening=0.10,
        face_lighting=0.10,
        debug=True,
        debug_dir="output/debug",
    )
    engine.save(result, "output/final.jpg")

    for stage in result.stages:
        state = "applied" if stage.changed else f"skipped: {stage.stats.get('reason')}"
        print(f"{stage.name:22s} {stage.strength:.2f}  {state}")

    print("\ndiagnostics:", json.dumps(result.diagnostics, indent=2))
    print(
        "\nLook at, in order:\n"
        "  output/debug/perception/02_mask_overview.jpg  what it decided the regions are\n"
        "  output/debug/difference_map.jpg               which pixels moved, amplified 12x\n"
        "  output/debug/before_after_face.jpg            the result, cropped to the face\n"
        "  output/debug/diffs/                           what each stage changed on its own"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/input/laughing_woman_with_teeth.jpg")
