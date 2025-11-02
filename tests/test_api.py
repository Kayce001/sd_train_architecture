from fastapi.testclient import TestClient

from lora_mvp.api.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
