# tests/test_app_v0_2.py

from fastapi.testclient import TestClient
from app_v0_2 import app  # 确保 app_v0_2.py 在根目录下

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_version" in body


def test_predict_valid_input():
    payload = {
        "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
        "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
        "s5": 0.02, "s6": -0.001
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert isinstance(result["prediction"], float)


def test_predict_invalid_input():
    # 缺少字段，或者类型错误（触发 Pydantic 校验失败）
    bad_payload = {
        "age": "invalid",  # 错误类型
        # 缺少其它字段
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # FastAPI 会返回 422
