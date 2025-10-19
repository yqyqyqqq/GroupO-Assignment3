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
├── train_v0_2.py
├── models/  
│   ├── feature_list.json
│   ├── metrics_v0_1.json
│   ├── metrics_v0_2.json
│   ├── model_v0_1.joblib
│   └── model_v0_2.joblib
├── .gitignore
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── Dockerfile
├── app_v0_1.py       
└── .github/  
    └── workflows/
        └── ci.yml

```

## Version 0.2 (Improved)

### A. Model Improvements
- **Algorithm:** RandomForestRegressor with hyperparameters
- **Preprocessing:** Improved feature scaling + seed control
- **Evaluation:** Better RMSE than v0.1 (see `CHANGELOG.md`)

### B. API Endpoints

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/predict` | POST | Predicts progression score from 10 input features |
| `/health` | GET | Returns status and model version info |

#### Example Request (`/predict`)
```json
{
  "age": 0.02,
  "sex": -0.044,
  "bmi": 0.06,
  "bp": -0.03,
  "s1": -0.02,
  "s2": 0.03,
  "s3": -0.02,
  "s4": 0.02,
  "s5": 0.02,
  "s6": -0.001
}
```

#### Example Response
```json
{
  "prediction": 174.9
}
```

### C. Local Development

```bash
# Train the model
python train_v0_2.py

# Run the FastAPI app
uvicorn app_v0_2:app --reload

# Run tests
pytest -v

# Manual API test (requires sample.json)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data "@sample.json"
```

## Project Structure
```
GROUP0-ASSIGNMENT3/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── models/
│   ├── feature_list.json
│   ├── metrics_v0_1.json
│   ├── metrics_v0_2.json
│   ├── model_v0_1.joblib
│   └── model_v0_2.joblib
├── tests/
│   └── test_train.py
├── app_v0_1.py
├── app_v0_2.py
├── train_v0_1.py
├── train_v0_2.py
├── requirements.txt
├── sample.json
├── Dockerfile
├── pytest.ini
├── README.md
└── CHANGELOG.md

