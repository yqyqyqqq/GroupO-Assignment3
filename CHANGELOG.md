#v0.2 – Model Improvement and Evaluation

##1. Summary of Changes

-Added RidgeRegression with α (regularization) tuning.
-Added RandomForestRegressor with tree-based hyperparameter tuning.
-Used same train/test split (random_state=42) for reproducibility.
-Logged metrics and models in /models/.

##2. Quantitative Comparison

| Version  | Model                 | RMSE ↓ | Main Parameters                                         | Comments                               |
| -------- | --------------------- | ------ | ------------------------------------------------------- | -------------------------------------- |
| v0.1     | LinearRegression      | 53.85  | StandardScaler + LinearRegression                       | Baseline                               |
| v0.2     | RidgeRegression       | 53.55  | α = 20.0, solver = auto                                 | Slightly lower error; less overfitting |
| v0.2(√)  | RandomForestRegressor | 52.87  | max_depth = 10, n_estimators = 400, max_features = sqrt | Best RMSE; captures non-linear effects |

##3. Discussion

Performance improved modestly (~1 RMSE point) but consistently across runs.
Ridge adds regularization stability, while Random Forest models non-linear feature interactions.
Even with small gains, the pipeline now includes automated tuning, logging, and full reproducibility.

##4. Reproducibility

-Fixed seed = 42, identical data split.
-Environment pinned via requirements.txt.
-Artifacts (model_v0_*.joblib, metrics_v0_*.json) saved for each iteration.

##5. Conclusion

v0.2 delivers a fully reproducible ML pipeline with clear documentation and hyperparameter optimization, justifying the selection of RandomForestRegressor as the final model for deployment.
