"""``portrait-retouch`` command line interface.

Example::

    portrait-retouch input.jpg \\
      --skin-cleanup 0.15 --skin-smoothing 0.20 --dark-circles 0.15 \\
      --teeth-whitening 0.10 --face-lighting 0.10 --debug

Writes::

    output/
      final.jpg
      debug/
        perception/  masks/  stages/  diffs/
        before_after.jpg  difference_map.jpg  report.json

Uses ``argparse`` rather than a CLI framework on purpose: one fewer dependency for a
five-flag interface.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from portrait_engine import __version__
from portrait_engine.core.pipeline import PortraitEngine
from portrait_engine.core.types import PortraitEngineError
from portrait_engine.perception.face_analyzer import AnalyzerConfig

STRENGTH_FLAGS: tuple[tuple[str, str, float], ...] = (
    ("--skin-cleanup", "skin_cleanup", 0.0),
    ("--skin-smoothing", "skin_smoothing", 0.0),
    ("--skin-tone", "skin_tone", 0.0),
    ("--dark-circles", "dark_circles", 0.0),
    ("--teeth-whitening", "teeth_whitening", 0.0),
    ("--face-lighting", "face_lighting", 0.0),
)

#: A sensible starting point, used by --preset natural.
PRESET_NATURAL = {
    "skin_cleanup": 0.15,
    "skin_smoothing": 0.20,
    "skin_tone": 0.20,
    "dark_circles": 0.15,
    "teeth_whitening": 0.10,
    "face_lighting": 0.10,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="portrait-retouch",
        description="Retouch a portrait with controlled, non-generative operations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Every strength is in [0, 1], where 1.0 is the maximum this engine considers\n"
            "natural, not the maximum the arithmetic allows. Tools left at 0 do not run."
        ),
    )
    parser.add_argument("input", help="path to a JPG or PNG portrait")
    parser.add_argument("-o", "--output-dir", default="output", help="default: output")
    parser.add_argument("--output-name", default="final.jpg")
    for flag, dest, default in STRENGTH_FLAGS:
        parser.add_argument(flag, dest=dest, type=float, default=default, metavar="0..1")
    parser.add_argument(
        "--preset", choices=("natural",),
        help="apply a starting point; explicit flags still override it",
    )
    parser.add_argument("--debug", action="store_true", help="write inspectable artifacts")
    parser.add_argument("--max-side", type=int, default=None, help="downscale the long side")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality, default 95")
    parser.add_argument(
        "--no-strict", action="store_true",
        help="disable the out-of-mask guardrail (for debugging only)",
    )
    parser.add_argument(
        "--no-diagnostics", action="store_true",
        help="skip identity-preservation measurement (saves one inference)",
    )
    parser.add_argument(
        "--use-parsing", action="store_true",
        help="enable the optional BiSeNet backend. Review the weight license first "
             "(see docs/THIRD_PARTY.md).",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--version", action="version", version=f"portrait-engine {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    strengths = dict(PRESET_NATURAL) if args.preset == "natural" else {}
    explicit = {dest: getattr(args, dest) for _, dest, _ in STRENGTH_FLAGS}
    strengths.update({k: v for k, v in explicit.items() if v > 0.0})

    if not strengths:
        print(
            "nothing to do: pass at least one strength, or --preset natural.\n"
            "Try: portrait-retouch input.jpg --preset natural --debug",
            file=sys.stderr,
        )
        return 2

    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = os.path.join(args.output_dir, "debug") if args.debug else None

    try:
        engine = PortraitEngine(analyzer_config=AnalyzerConfig(use_parsing=args.use_parsing))
        started = time.perf_counter()
        result = engine.process(
            args.input,
            debug=args.debug,
            debug_dir=debug_dir,
            max_side=args.max_side,
            strict=not args.no_strict,
            run_diagnostics=not args.no_diagnostics,
            **strengths,
        )
        elapsed = time.perf_counter() - started
        output_path = engine.save(
            result, os.path.join(args.output_dir, args.output_name), quality=args.quality
        )
    except PortraitEngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = result.report()
    report["elapsed_seconds"] = round(elapsed, 3)
    report["output"] = output_path

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    print(f"{args.input} -> {output_path}  ({elapsed:.2f}s)")
    print(f"face: {result.analysis.bbox.width}x{result.analysis.bbox.height}px  "
          f"interocular {result.analysis.geometry.interocular_px:.0f}px  "
          f"backend {result.analysis.parsing_backend}")
    for stage in result.stages:
        state = "applied" if stage.changed else f"skipped ({stage.stats.get('reason', '')})"
        print(f"  {stage.name:22s} {stage.strength:.2f}  {state}")
    if result.diagnostics:
        diag = result.diagnostics
        print(f"  texture retention {diag['texture_retention']:.3f}   "
              f"out-of-region leak {diag['out_of_region_leak']['max_levels']:.2f}/255   "
              f"landmark drift {diag.get('landmark_drift', {}).get('rms')}")
        for warning in diag.get("warnings", []):
            print(f"  WARNING: {warning}", file=sys.stderr)
    if args.debug:
        print(f"  debug artifacts in {debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
