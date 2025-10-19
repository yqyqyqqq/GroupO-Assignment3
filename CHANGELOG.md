## 1. Summary of Changes

- Tested Ridge Regression and Random Forest models.    
- Selected Random Forest as the final model after tuning.        
- Logged metrics and models in /models/.      

## 2. Results

| Version  | Model                 | RMSE ↓ | Main Parameters                                         |  
| -------- | --------------------- | ------ | ------------------------------------------------------- |  
| v0.1     | LinearRegression      | 53.85  | StandardScaler + LinearRegression                       |  
| v0.2     | RidgeRegression       | 53.55  | α = 20.0, solver = auto                                 |  
| v0.2(√)  | RandomForestRegressor | 52.87  | max_depth = 10, n_estimators = 400, max_features = sqrt | 

## 3. Discussion

- The Ridge Regression showed a small improvement by reducing overfitting compared to the baseline.  
- The Random Forest performed better than Ridge Regression by capturing non-linear relationships in the data.  
  
## 4. Reproducibility

- Fixed seed = 42, same data split.  
- Environment pinned via requirements.txt.  
- Artifacts (model_v0_*.joblib, metrics_v0_*.json) saved for each iteration.  
