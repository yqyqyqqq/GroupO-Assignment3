# Virtual Diabetes Clinic Triage Service

This is a machine learning-based API service designed to predict the short-term disease progression risk for diabetes patients. The service is packaged as a Docker image and is built and released through a fully automated CI/CD pipeline using GitHub Actions.

---

## 📋 Iteration Plan & Versions

This project was completed in two iterations as per the assignment requirements:

### v0.1 (Baseline Model)

- **Model:** LinearRegression
- **Goal:** To establish a fully functional, working API service baseline.
- **Image Tag:** `:v0.1`

### v0.2 (Improved Model)

- **Model:** Random Forest
- **Improvement:** The Random Forest performed better than Ridge Regression by capturing non-linear relationships in the data..
- **Image Tag:** `:v0.2`

> **Note:** For detailed changes, the rationale behind improvements, and a side-by-side performance comparison (RMSE) between v0.1 and v0.2, please see the `CHANGELOG.md` file.

---

## 🚀 How to Run the Service

The service is automatically published as a Docker image via GitHub Actions. You can run any version of the service locally by following these steps:

### Pull the Docker Image

Replace `<username>` and `<repository>` with your actual GitHub username and repository name. Replace `:v0.1` with the version you wish to run (e.g., `:v0.2`).

```bash
docker pull ghcr.io/<username>/<repository>:v0.1
```

### Run the Docker Container

```bash
docker run -d -p 8000:8000 --name clinic-service ghcr.io/<username>/<repository>:v0.1
```

**Parameters:**
- `-d`: Runs the container in detached mode.
- `-p 8000:8000`: Maps port 8000 on your local machine to port 8000 inside the container.
- `--name clinic-service`: Assigns a name to the running container.

🎉 **The service is now running on http://localhost:8000!**

---

## ⚙️ API Usage

The service provides two main API endpoints:

### Health Check (`/health`)

Checks if the service is running correctly and returns the currently loaded model version.

- **URL:** `GET /health`

**Example Request:**

```bash
curl http://localhost:8000/health
```

**Success Response (for v0.1):**

```json
{
  "status": "ok",
  "model_version": "0.1"
}
```

---

### Disease Progression Prediction (`/predict`)

Accepts patient feature data and returns a continuous disease progression risk score.

- **URL:** `POST /predict`

**Request Body:**

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

**Example Request (Linux/macOS/Git Bash):**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Example Request (Windows PowerShell):**

```powershell
curl -Method POST -Uri "http://localhost:8000/predict" `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{
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
  }'
```

**Success Response:**

```json
{
  "prediction": 152.133
}
```

---

## 📂 Project Structure

```
.
├── .github/workflows/          # GitHub Actions CI/CD Workflows
│   ├── ci.yml                  # CI pipeline for linting and smoke tests
│   └── release.yml             # Release pipeline for building and publishing
├── src/                        # All Python source code
│   ├── main.py                 # FastAPI service
│   └── train.py                # Model training script
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── CHANGELOG.md                # Log of changes for each version
├── Dockerfile                  # Instructions to build the Docker image
├── README.md                   # This project's documentation
└── requirements.txt            # List of Python dependencies
```

---

## 🤖 CI/CD Automation Workflow

This project utilizes GitHub Actions for MLOps automation:

### CI (Continuous Integration)

**Triggered on:** Every push to the `main` branch or when a Pull Request is created.

The `ci.yml` workflow runs:
- Linting checks
- Training smoke tests

This ensures code quality and integrity.

### Release (Continuous Deployment)

**Triggered on:** When a tag in the format `v*` (e.g., `v0.1`, `v0.2`) is pushed to the repository.

The `release.yml` workflow automates the entire release process:
1. Training the model
2. Building the Docker image
3. Running container tests
4. Pushing the image to GHCR (GitHub Container Registry)
5. Creating a GitHub Release with metrics

---

## 📝 License

This project is part of an educational assignment.

## 🤝 Contributing

Everymember in this group makes a great contribution!
Yuanqing Li,
Qingxiu Zeng,
Nanxi Li,
Wentin Fang,

---

**Made with ❤️ for better diabetes care**
