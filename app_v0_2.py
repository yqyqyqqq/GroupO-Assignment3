from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# 加载模型
model = joblib.load("models/model_v0_2.joblib")


class Features(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v0.2"}


@app.post("/predict")
def predict(features: Features):
    try:
        X = np.array([[features.age, features.sex, features.bmi, features.bp,
                       features.s1, features.s2, features.s3, features.s4,
                       features.s5, features.s6]])
        prediction = float(model.predict(X)[0])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
