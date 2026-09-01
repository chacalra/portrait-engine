"""Batch evaluation harness.

Processes every image in a folder, writes a side-by-side comparison and a difference map for
each, and collects the numeric diagnostics into a CSV plus a summary. This is the tool for
judging a parameter change across a set of portraits instead of one favourite photo.

Usage::

    python scripts/evaluate_folder.py input/ output/
    python scripts/evaluate_folder.py input/ output/ --preset natural --max-side 1600
    python scripts/evaluate_folder.py input/ output/ --skin-smoothing 0.4 --strength-sweep
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
import traceback

import cv2
import numpy as np

from portrait_engine import PortraitEngine, PortraitEngineError, load_image
from portrait_engine.debug import visualize as viz

EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

PRESET_NATURAL = {
    "skin_cleanup": 0.15,
    "skin_smoothing": 0.20,
    "dark_circles": 0.15,
    "teeth_whitening": 0.10,
    "face_lighting": 0.10,
}

CSV_FIELDS = [
    "file", "status", "width", "height", "interocular_px", "elapsed_s",
    "texture_retention", "leak_max_levels", "landmark_drift_rms",
    "changed_pixel_fraction", "stages_applied", "stages_skipped", "warnings", "note",
]


def find_images(folder: str) -> list[str]:
    found: list[str] = []
    for pattern in EXTENSIONS:
        found.extend(glob.glob(os.path.join(folder, pattern)))
    return sorted(set(found))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--preset", choices=("natural",), default="natural")
    parser.add_argument("--skin-cleanup", type=float, default=None)
    parser.add_argument("--skin-smoothing", type=float, default=None)
    parser.add_argument("--dark-circles", type=float, default=None)
    parser.add_argument("--teeth-whitening", type=float, default=None)
    parser.add_argument("--face-lighting", type=float, default=None)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--debug", action="store_true", help="write full debug output per image")
    parser.add_argument(
        "--strength-sweep", action="store_true",
        help="also render a 0 / 0.25 / 0.5 / 1.0 strip per image, for judging the response curve",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    strengths = dict(PRESET_NATURAL)
    for key in list(strengths):
        value = getattr(args, key)
        if value is not None:
            strengths[key] = value

    images = find_images(args.input_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"no images found in {args.input_dir}")
        return 1

    for sub in ("final", "compare", "difference"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)
    if args.strength_sweep:
        os.makedirs(os.path.join(args.output_dir, "sweep"), exist_ok=True)

    engine = PortraitEngine()
    rows: list[dict[str, object]] = []

    print(f"{len(images)} images, strengths {strengths}")
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        row: dict[str, object] = dict.fromkeys(CSV_FIELDS, "")
        row["file"] = os.path.basename(path)
        started = time.perf_counter()
        try:
            image = load_image(path, max_side=args.max_side)
            row["width"], row["height"] = image.width, image.height
            result = engine.process(
                image,
                debug=args.debug,
                debug_dir=os.path.join(args.output_dir, "debug", name) if args.debug else None,
                **strengths,
            )
            elapsed = time.perf_counter() - started

            engine.save(result, os.path.join(args.output_dir, "final", f"{name}.jpg"))
            comparison = viz.before_after(image.rgb, result.image)
            cv2.imwrite(
                os.path.join(args.output_dir, "compare", f"{name}.jpg"),
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(args.output_dir, "difference", f"{name}.jpg"),
                cv2.cvtColor(viz.difference_map(image.rgb, result.image, gain=14.0), cv2.COLOR_RGB2BGR),
            )
            if args.strength_sweep:
                _write_sweep(engine, image, result, os.path.join(args.output_dir, "sweep", f"{name}.jpg"))

            diag = result.diagnostics
            applied = [s.name for s in result.stages if s.changed]
            skipped = [s.name for s in result.stages if not s.changed]
            row.update(
                status="ok",
                interocular_px=round(result.analysis.geometry.interocular_px, 1),
                elapsed_s=round(elapsed, 3),
                texture_retention=diag.get("texture_retention", ""),
                leak_max_levels=diag.get("out_of_region_leak", {}).get("max_levels", ""),
                landmark_drift_rms=(diag.get("landmark_drift") or {}).get("rms", ""),
                changed_pixel_fraction=diag.get("changed_pixel_fraction", ""),
                stages_applied=";".join(applied),
                stages_skipped=";".join(skipped),
                warnings=" | ".join(diag.get("warnings", [])),
            )
            print(f"  ok    {row['file']:44s} {elapsed:5.2f}s  texture={row['texture_retention']} "
                  f"skipped={skipped or '-'}")
        except PortraitEngineError as exc:
            row.update(status="rejected", note=str(exc))
            print(f"  skip  {row['file']:44s} {exc}")
        except Exception as exc:
            row.update(status="error", note=f"{type(exc).__name__}: {exc}")
            print(f"  ERROR {row['file']:44s} {type(exc).__name__}: {exc}")
            traceback.print_exc()
        rows.append(row)

    csv_path = os.path.join(args.output_dir, "evaluation.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    summary = {
        "images": len(rows),
        "processed": len(ok),
        "rejected": sum(1 for r in rows if r["status"] == "rejected"),
        "errors": sum(1 for r in rows if r["status"] == "error"),
        "strengths": strengths,
        "median_elapsed_s": round(float(np.median([r["elapsed_s"] for r in ok])), 3) if ok else None,
        "median_texture_retention": round(
            float(np.median([r["texture_retention"] for r in ok])), 4
        ) if ok else None,
        "max_leak_levels": round(max([float(r["leak_max_levels"]) for r in ok]), 4) if ok else None,
        "images_with_warnings": [r["file"] for r in ok if r["warnings"]],
    }
    with open(os.path.join(args.output_dir, "evaluation.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + json.dumps(summary, indent=2))
    print(f"\n{csv_path}")
    return 0 if summary["errors"] == 0 else 1


def _write_sweep(engine: PortraitEngine, image, result, path: str) -> None:
    """A single strip showing the same face at increasing strength, cropped to the face."""
    from portrait_engine.debug import visualize as viz

    tiles = []
    labels = []
    for strength in (0.0, 0.25, 0.5, 1.0):
        if strength == 0.0:
            rgb = image.rgb
        else:
            swept = engine.process(
                image,
                analysis=result.analysis,
                run_diagnostics=False,
                skin_cleanup=strength,
                skin_smoothing=strength,
                dark_circles=strength,
                teeth_whitening=strength,
                face_lighting=strength,
            )
            rgb = swept.image
        tiles.append(viz.crop_zoom(viz._u8(rgb), result.analysis, "face"))
        labels.append(f"{strength:.2f}")

    height = min(tile.shape[0] for tile in tiles)
    width = min(tile.shape[1] for tile in tiles)
    tiles = [tile[:height, :width] for tile in tiles]
    strip = np.hstack(tiles)
    scale = max(0.5, strip.shape[1] / 2600.0)
    for index, label in enumerate(labels):
        position = (index * width + 12, int(34 * scale))
        cv2.putText(strip, label, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(strip, label, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    raise SystemExit(main())
