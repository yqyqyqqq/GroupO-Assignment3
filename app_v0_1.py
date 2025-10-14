import joblib
import json

# 加载模型
model = joblib.load('models/model_v0_1.joblib')

# 加载特征顺序
with open('models/feature_list.json', 'r') as f:
    feature_list = json.load(f)

print(f"特征顺序: {feature_list}")
# 输出: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

### 3️⃣ **实现 FastAPI 预测接口**
from fastapi import FastAPI
import joblib
import json
import numpy as np

app = FastAPI()

# 加载模型和特征列表
model = joblib.load('models/model_v0_1.joblib')
with open('models/feature_list.json', 'r') as f:
    feature_list = json.load(f)

@app.post("/predict")
def predict(data: dict):
    """
    输入示例:
    {
        "age": 0.038076,
        "sex": 0.050680,
        "bmi": 0.061696,
        "bp": 0.021872,
        "s1": -0.044223,
        "s2": -0.034821,
        "s3": -0.043401,
        "s4": -0.002592,
        "s5": 0.019908,
        "s6": -0.017646
    }
    """
    # 按照 feature_list 顺序提取特征
    features = [data[f] for f in feature_list]
    features_array = np.array([features])
    
    # 预测
    prediction = model.predict(features_array)[0]
    
    return {
        "prediction": float(prediction),
        "model_version": "v0.1"
    }
