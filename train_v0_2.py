from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
import numpy as np
import joblib
import json
import os

# 1. 加载数据
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 定义基础模型
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# 4. 定义参数搜索分布（随机搜索的核心）
param_dist = {
    "n_estimators": randint(100, 500),        # 随机采样 100~500 棵树
    "max_depth": [5, 8, 10, 12, 15, None],   # 离散选项
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2"]
}

# 5. 随机搜索（迭代次数可根据时间调节）
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,  # 搜索 20 组参数，比 GridSearch 快得多
    cv=3,
    n_jobs=-1,
    random_state=42,
    scoring="neg_root_mean_squared_error",
    verbose=1
)

# 6. 训练模型
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
best_params = random_search.best_params_

# 7. 评估最优模型
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Best Parameters (Randomized Search):", best_params)
print(f"RandomForest RMSE: {rmse:.4f}")

# 8. 保存模型与指标
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model_v0_2_rf_random.joblib")

metrics = {
    "rmse": float(rmse),
    "best_params": best_params,
    "model_type": "RandomForestRegressor_random"
}
with open("models/metrics_v0_2_rf_random.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("随机搜索版本的 RandomForest 模型已训练完成并保存到 models/ 文件夹。")