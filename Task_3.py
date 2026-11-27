import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Import your FastAPI app from Task_2.py
from Task_2 import app

client = TestClient(app)


def make_mock_gemini_response(payload_text: str, prompt_tokens=100, completion_tokens=50):
    """Helper to mimic the Gemini SDK response structure."""
    return SimpleNamespace(
        text=payload_text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
            total_token_count=prompt_tokens + completion_tokens,
        ),
    )


VALID_LLM_JSON = json.dumps({
    "candidate_name": "Jane Doe",
    "score": 85,
    "matched_skills": ["Python", "FastAPI"],
    "missing_skills": ["Docker"],
    "reasoning": "Strong Python and FastAPI background; missing Docker.",
})


# 1. Test Success Case
@patch("Task_2.client.models.generate_content")
def test_analyze_success(mock_generate):
    mock_generate.return_value = make_mock_gemini_response(VALID_LLM_JSON)

    response = client.post(
        "/analyze",
        data={
            "job_description": "Need Python, FastAPI, and Docker developer.",
            "resume_text": "Jane Doe. Experienced in Python and FastAPI.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["candidate_name"] == "Jane Doe"
    assert body["score"] == 85
    assert "Python" in body["matched_skills"]
    assert "request_id" in body
    assert mock_generate.call_count == 1


# 2. Test Invalid JSON on Attempt 1, then Valid JSON on Retry (Attempt 2)
@patch("Task_2.client.models.generate_content")
def test_analyze_invalid_json_then_success(mock_generate):
    # First call returns broken JSON; second call returns valid JSON
    mock_generate.side_effect = [
        make_mock_gemini_response("{invalid_json: true"),
        make_mock_gemini_response(VALID_LLM_JSON),
    ]

    response = client.post(
        "/analyze",
        data={
            "job_description": "Need Python, FastAPI, and Docker developer.",
            "resume_text": "Jane Doe. Experienced in Python and FastAPI.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["candidate_name"] == "Jane Doe"
    assert body["score"] == 85
    # Verify retry happened automatically
    assert mock_generate.call_count == 2


# 3. Test Empty File Upload
def test_analyze_empty_file():
    empty_txt = io.BytesIO(b"   ")  # Whitespace-only / empty text file

    response = client.post(
        "/analyze",
        data={"job_description": "Need Python developer."},
        files={"resume_file": ("empty_resume.txt", empty_txt, "text/plain")},
    )

    assert response.status_code == 422
    assert "Could not extract readable text" in response.json()["detail"]


# 4. Test Wrong File Type Upload
def test_analyze_wrong_file_type():
    fake_image = io.BytesIO(b"\x89PNG\r\n\x1a\nfake_image_bytes")

    response = client.post(
        "/analyze",
        data={"job_description": "Need Python developer."},
        files={"resume_file": ("photo.png", fake_image, "image/png")},
    )

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


# 5. Test Health Endpoint (boosts test coverage)
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
