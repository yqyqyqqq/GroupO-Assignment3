# train_v0_2.py
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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

# 3. 定义基础模型
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# 4. 参数搜索网格
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

# 5. 网格搜索
grid_search = GridSearchCV(
    rf, param_grid, cv=3, n_jobs=-1, scoring="neg_root_mean_squared_error", verbose=1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# 6. 评估最优模型
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Best Parameters:", best_params)
print(f"Improved RandomForest RMSE: {rmse:.4f}")

# 7. 保存模型与指标
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model_v0_2.joblib")

metrics = {
    "rmse": float(rmse),
    "best_params": best_params,
    "model_type": "RandomForestRegressor"
}
with open("models/metrics_v0_2.json", "w") as f:
    json.dump(metrics, f)

print("调参后的 RandomForest 模型已训练完成并保存。")