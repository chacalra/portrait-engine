"""Shared fixtures.

Tests that need a real photograph are marked ``slow`` and skip cleanly when the evaluation
fixtures are not present, so ``pytest`` still passes on a fresh clone. Run
``python scripts/fetch_test_images.py`` to get them.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(REPO_ROOT, "data", "input")

#: A frontal portrait with visible teeth, used as the default subject.
PRIMARY = "laughing_woman_with_teeth.jpg"


def _portraits() -> list[str]:
    return sorted(
        path for path in glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
        if os.path.getsize(path) > 1024
    )


def require_portrait(name: str = PRIMARY) -> str:
    path = os.path.join(INPUT_DIR, name)
    if not os.path.isfile(path):
        pytest.skip(f"missing test fixture {path}; run scripts/fetch_test_images.py")
    return path


@pytest.fixture(scope="session")
def portrait_path() -> str:
    return require_portrait()


@pytest.fixture(scope="session")
def portrait_paths() -> list[str]:
    paths = _portraits()
    if not paths:
        pytest.skip("no test fixtures; run scripts/fetch_test_images.py")
    return paths


@pytest.fixture(scope="session")
def analyzer():
    from portrait_engine import FaceAnalyzer

    return FaceAnalyzer()


@pytest.fixture(scope="session")
def portrait(portrait_path: str):
    from portrait_engine import load_image

    return load_image(portrait_path, max_side=900)


@pytest.fixture(scope="session")
def analysis(analyzer, portrait):
    return analyzer.analyze(portrait)


@pytest.fixture(scope="session")
def engine():
    from portrait_engine import PortraitEngine

    return PortraitEngine()


@pytest.fixture(scope="session")
def two_face_image(portrait_paths: list[str]):
    """A synthetic two-person image, built by placing two different portraits side by side.

    Used to assert the MVP's single-subject policy. Synthesizing it is preferable to shipping a
    photograph of two identifiable people just for a test.
    """
    from portrait_engine import from_array, load_image

    if len(portrait_paths) < 2:
        pytest.skip("need at least two fixtures to build a two-face image")
    left = load_image(portrait_paths[0], max_side=700).rgb
    right = load_image(portrait_paths[1], max_side=700).rgb
    height = min(left.shape[0], right.shape[0])
    return from_array(np.hstack([left[:height], right[:height]]))


@pytest.fixture
def synthetic_no_face():
    """A gradient with noise. No face, decodable, correct dtype."""
    from portrait_engine import from_array

    rng = np.random.default_rng(0)
    ramp = np.linspace(0.15, 0.75, 320, dtype=np.float32)
    base = np.repeat(ramp[None, :], 240, axis=0)
    rgb = np.stack([base, base * 0.9, base * 0.8], axis=-1)
    rgb = np.clip(rgb + rng.normal(0, 0.02, rgb.shape).astype(np.float32), 0, 1)
    return from_array(rgb)
