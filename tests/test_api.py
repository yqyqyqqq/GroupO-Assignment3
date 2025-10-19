# tests/test_api.py
import os
import pytest
from fastapi.testclient import TestClient
from src.main import app

# 在运行测试前设置环境变量，模拟 Docker 环境
os.environ["MODEL_VERSION"] = "0.1"

# 创建一个测试客户端
client = TestClient(app)

def test_health_check_success():
    """测试 /health 接口能否成功返回"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "0.1"}

def test_predict_success():
    """测试 /predict 接口在输入正确时能否成功预测"""
    payload = {
        "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
        "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
        "s5": 0.02, "s6": -0.001
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)

def test_predict_bad_input():
    """测试 /predict 接口在输入错误时能否返回正确的错误码"""
    # 故意缺少一些字段
    payload = {
        "age": 0.02, "sex": -0.044, "bmi": 0.06
    }
    response = client.post("/predict", json=payload)
    # FastAPI/Pydantic 的验证错误会返回 422
    assert response.status_code == 422
