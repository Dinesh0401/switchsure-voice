from fastapi.testclient import TestClient
from switchsure_api import create_app


def client() -> TestClient:
    return TestClient(create_app())


def test_health_has_request_id() -> None:
    response = client().get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "switchsure-api"}
    assert response.headers["X-Request-ID"]


def test_demo_analyzes_uncertain_code_switched_dosage() -> None:
    response = client().post(
        "/v1/demo/analyze-evidence",
        json={
            "model_version": "fixture-v1",
            "words": [
                {
                    "surface": "மருந்து",
                    "start_ms": 0,
                    "end_ms": 350,
                    "decoder_confidence": 0.92,
                    "boundary_probability": 0.25,
                    "languages": [
                        {"code": "ta", "probability": 0.94},
                        {"code": "en", "probability": 0.06},
                    ],
                },
                {
                    "surface": "500",
                    "start_ms": 360,
                    "end_ms": 580,
                    "decoder_confidence": 0.22,
                    "acoustic_quality": 0.6,
                    "boundary_probability": 0.52,
                    "languages": [
                        {"code": "neutral", "probability": 0.55},
                        {"code": "en", "probability": 0.45},
                    ],
                },
                {
                    "surface": "mg",
                    "start_ms": 590,
                    "end_ms": 760,
                    "decoder_confidence": 0.35,
                    "acoustic_quality": 0.65,
                    "boundary_probability": 0.49,
                    "languages": [
                        {"code": "en", "probability": 0.85},
                        {"code": "ta", "probability": 0.15},
                    ],
                },
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] in {"review", "defer"}
    assert body["calibration_status"] == "uncalibrated_baseline"
    assert body["entities"][0]["entity_type"] == "dosage"
    assert body["entities"][0]["surface"] == "500 mg"


def test_demo_rejects_non_normalized_language_probabilities() -> None:
    response = client().post(
        "/v1/demo/analyze-evidence",
        json={
            "words": [
                {
                    "surface": "hello",
                    "start_ms": 0,
                    "end_ms": 100,
                    "decoder_confidence": 0.9,
                    "languages": [{"code": "en", "probability": 0.8}],
                }
            ]
        },
    )

    assert response.status_code == 422
