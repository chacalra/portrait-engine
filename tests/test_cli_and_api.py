"""The two adapters. Both must be thin: they parse, call the engine, and serialize."""

from __future__ import annotations

import json

import pytest

from portrait_engine.cli import build_parser, main

# --------------------------------------------------------------------------- CLI


def test_parser_defaults_are_all_zero():
    args = build_parser().parse_args(["input.jpg"])
    for dest in ("skin_cleanup", "skin_smoothing", "dark_circles", "teeth_whitening", "face_lighting"):
        assert getattr(args, dest) == 0.0


def test_cli_without_any_strength_refuses_politely(portrait_path, tmp_path, capsys):
    code = main([portrait_path, "-o", str(tmp_path)])
    assert code == 2
    assert "nothing to do" in capsys.readouterr().err


def test_cli_runs_the_documented_invocation(portrait_path, tmp_path, capsys):
    """The exact command from the brief."""
    code = main([
        portrait_path, "-o", str(tmp_path), "--max-side", "700",
        "--skin-cleanup", "0.15", "--skin-smoothing", "0.20", "--dark-circles", "0.15",
        "--teeth-whitening", "0.10", "--face-lighting", "0.10", "--debug",
    ])
    assert code == 0
    assert (tmp_path / "final.jpg").is_file()
    assert (tmp_path / "debug" / "before_after.jpg").is_file()
    assert (tmp_path / "debug" / "difference_map.jpg").is_file()
    assert (tmp_path / "debug" / "masks").is_dir()
    assert (tmp_path / "debug" / "stages").is_dir()
    out = capsys.readouterr().out
    assert "skin_smoothing" in out and "texture retention" in out


def test_cli_preset(portrait_path, tmp_path):
    assert main([portrait_path, "-o", str(tmp_path), "--max-side", "600", "--preset", "natural"]) == 0
    assert (tmp_path / "final.jpg").is_file()


def test_cli_explicit_flag_overrides_the_preset(portrait_path, tmp_path, capsys):
    main([
        portrait_path, "-o", str(tmp_path), "--max-side", "600",
        "--preset", "natural", "--skin-smoothing", "0.9", "--json",
    ])
    report = json.loads(capsys.readouterr().out)
    smoothing = next(s for s in report["stages"] if s["name"] == "skin_smoothing")
    assert smoothing["strength"] == pytest.approx(0.9)


def test_cli_json_output_is_parseable(portrait_path, tmp_path, capsys):
    assert main([portrait_path, "-o", str(tmp_path), "--max-side", "600",
                 "--skin-smoothing", "0.2", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["output"].endswith("final.jpg")
    assert report["elapsed_seconds"] > 0
    assert report["face"]["landmark_count"] == 478


def test_cli_reports_no_face_as_a_clean_failure(tmp_path, capsys, synthetic_no_face):
    from portrait_engine import save_image

    path = tmp_path / "noface.png"
    save_image(synthetic_no_face, str(path))
    code = main([str(path), "-o", str(tmp_path / "out"), "--skin-smoothing", "0.2"])
    assert code == 1
    assert "no face detected" in capsys.readouterr().err


def test_cli_reports_a_bad_file_as_a_clean_failure(tmp_path, capsys):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"garbage" * 50)
    assert main([str(path), "-o", str(tmp_path / "out"), "--skin-smoothing", "0.2"]) == 1
    assert "error:" in capsys.readouterr().err


# --------------------------------------------------------------------------- FastAPI


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2", reason="starlette's TestClient needs it")
    from fastapi.testclient import TestClient

    from portrait_engine.api.app import app

    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_tools_endpoint_serves_the_contracts(client):
    schemas = client.get("/tools").json()
    assert {schema["name"] for schema in schemas} == {
        "skin_cleanup", "skin_smoothing", "equalize_skin_tone", "reduce_dark_circles",
        "whiten_teeth", "balance_face_lighting",
    }
    for schema in schemas:
        assert schema["parameters"]["properties"]["strength"]["maximum"] == 1.0


def test_analyze_endpoint(client, portrait_path):
    with open(portrait_path, "rb") as handle:
        response = client.post("/analyze", files={"file": ("p.jpg", handle, "image/jpeg")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["landmark_count"] == 478
    assert payload["bbox"]["width"] > 0
    assert "skin" in payload["masks"]


def test_retouch_endpoint_returns_a_jpeg_and_a_report(client, portrait_path):
    with open(portrait_path, "rb") as handle:
        response = client.post(
            "/retouch",
            files={"file": ("p.jpg", handle, "image/jpeg")},
            data={"skin_smoothing": 0.2, "teeth_whitening": 0.1},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"
    report = json.loads(response.headers["X-Portrait-Report"])
    assert [s["name"] for s in report["stages"]] == ["skin_smoothing", "whiten_teeth"]


def test_retouch_rejects_an_all_zero_request(client, portrait_path):
    with open(portrait_path, "rb") as handle:
        response = client.post("/retouch", files={"file": ("p.jpg", handle, "image/jpeg")})
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "no_operation"


def test_retouch_maps_no_face_to_422(client, tmp_path, synthetic_no_face):
    from portrait_engine import save_image

    path = tmp_path / "noface.png"
    save_image(synthetic_no_face, str(path))
    with open(path, "rb") as handle:
        response = client.post(
            "/retouch", files={"file": ("p.png", handle, "image/png")},
            data={"skin_smoothing": 0.2},
        )
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "no_face_detected"


def test_retouch_maps_a_bad_upload_to_400(client):
    response = client.post(
        "/retouch", files={"file": ("p.jpg", b"not an image", "image/jpeg")},
        data={"skin_smoothing": 0.2},
    )
    assert response.status_code == 400


def test_retouch_rejects_an_out_of_range_strength(client, portrait_path):
    with open(portrait_path, "rb") as handle:
        response = client.post(
            "/retouch", files={"file": ("p.jpg", handle, "image/jpeg")},
            data={"skin_smoothing": 1.5},
        )
    assert response.status_code == 422
