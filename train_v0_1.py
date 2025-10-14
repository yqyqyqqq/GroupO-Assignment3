# train_v0_1.py
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import json
import os

# 1. 加载数据
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 训练模型（LinearRegression）
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 5. 模型评估
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# 6. 保存模型与特征
os.makedirs("models", exist_ok=True)
joblib.dump({"scaler": scaler, "model": model}, "models/model_v0_1.joblib")

with open("models/feature_list.json", "w") as f:
    json.dump(list(X.columns), f)

# 7. 保存指标
metrics = {"rmse": float(rmse)}
with open("models/metrics_v0_1.json", "w") as f:
    json.dump(metrics, f)

print("模型已训练完成并保存到 models/ 文件夹。")
