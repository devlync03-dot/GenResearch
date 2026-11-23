from fastapi.testclient import TestClient

import main


def test_readiness_returns_success_when_generation_and_embedding_are_available(monkeypatch):
    async def ready():
        return {
            "status": "ok",
            "available_models": [
                "modal:deepseek-ai/DeepSeek-V4.1-Flash-1017",
                "ollama:nomic-embed-text",
            ],
            "missing_models": [],
            "errors": [],
        }

    monkeypatch.setattr(main, "check_provider_readiness", ready)

    response = TestClient(main.app).get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_returns_unavailable_when_a_configured_provider_is_missing(monkeypatch):
    async def not_ready():
        return {
            "status": "not_ready",
            "available_models": ["modal:deepseek-ai/DeepSeek-V4.1-Flash-1017"],
            "missing_models": ["ollama:nomic-embed-text"],
            "errors": [],
        }

    monkeypatch.setattr(main, "check_provider_readiness", not_ready)

    response = TestClient(main.app).get("/health/ready")

    assert response.status_code == 503
    assert response.json()["missing_models"] == ["ollama:nomic-embed-text"]
