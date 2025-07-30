# Churn Prediction MLOps Pipeline

## Overview
This project implements an end-to-end MLOps pipeline for customer churn prediction using machine learning. It covers data preprocessing, model training, prediction, API serving, containerization, CI/CD automation, and Kubernetes deployment.

## Features
- Data preprocessing and feature engineering
- Model training with RandomForestClassifier
- Prediction script for batch inference
- FastAPI REST API for real-time predictions
- Dockerfile for containerization
- GitHub Actions workflow for CI/CD
- Kubernetes manifests for scalable deployment
- Unit test integration

## Project Structure
```
├── data/
│   └── raw/
├── models/
├── notebooks/
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   └── serve_api.py
├── tests/
│   ├── __init__.py
│   └── test_dummy.py
├── requirements.txt
├── dockerfile/
│   └── Dockerfile
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
└── .github/
    └── workflows/
        └── train.yml
```

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess data
```bash
python src/preprocess.py
```

### 3. Train the model
```bash
python src/train_model.py
```

### 4. Run batch predictions
```bash
python src/predict.py
```

### 5. Serve API
```bash
uvicorn src.serve_api:app --reload
```
Visit [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

### 6. Build and run Docker container
```bash
docker build -t churn-api -f dockerfile/Dockerfile .
docker run -p 8000:8000 churn-api
```

### 7. Deploy to Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## CI/CD
- Automated with GitHub Actions (`.github/workflows/train.yml`)
- Runs preprocessing, training, tests, and uploads model artifact on push

## Testing
- Unit tests are located in the `tests/` directory
- Run all tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

## License
This project uses only open source libraries and is free to use, modify, and distribute.

## Author
Sri0401
