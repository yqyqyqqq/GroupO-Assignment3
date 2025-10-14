# Virtual Diabetes Clinic Triage (v0.1 Baseline)

This project simulates a machine learning service that predicts short-term diabetes progression using the open scikit-learn diabetes dataset.

## Version: v0.1 (Baseline)

### A. Model Details
- Algorithm: StandardScaler + LinearRegression
- Dataset: sklearn.datasets.load_diabetes
- Metric (RMSE): see models/metrics_v0_1.json

### B. Output Files
| File | Description |
|------|--------------|
| models/model_v0_1.joblib | Trained model (scaler + regressor) |
| models/feature_list.json | Feature names used during training |  
| models/metrics_v0_1.json | Evaluation metrics (RMSE, etc.) |

### C. How to Run Training
```bash
python train_v0_1.py
```

### D. Handoff to Developer B
Developer B should:
1. Load the model from models/model_v0_1.joblib
2. Use feature order from models/feature_list.json
3. Implement FastAPI service with this model

## Project Structure
```
diabetes-rtds/         
├── train_v0_1.py
├── models/  
│   ├── model_v0_1.joblib
│   ├── feature_list.json
│   └── metrics_v0_1.json  
├── .gitignore
└── README.md
├── requirements.txt
├── Dockerfile
├── app_v0_1.py       
└── .github/  
    └── workflows/
        └── ci.yml
