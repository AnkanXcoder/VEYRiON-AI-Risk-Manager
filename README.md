# AI Risk Manager

AI Risk Manager is a defense-only, cost-sensitive fraud-risk decision prototype built for Razorpay Hackathon Track 02. It uses a public credit-card fraud benchmark and is not production-ready.

## Problem and solution

The project helps a merchant route a transaction to **APPROVE**, **REVIEW**, or **BLOCK** based on an XGBoost fraud probability. It prioritizes the trade-off between fraud capture and legitimate-customer friction rather than maximizing recall alone.

## Model and policy

- Model: XGBoost
- Inputs: `Time`, `V1`–`V28`, and `Amount` (30 features)
- APPROVE: probability below 5%
- REVIEW: probability from 5% to below 20%
- BLOCK: probability of 20% or higher
- Prototype cost assumption: false negative : false positive = 10 : 1

The cost ratio is illustrative only; it is not a claim about Razorpay or merchant losses. Threshold selection uses validation data only, and held-out test data is used solely for final evaluation.

## API

`POST /predict` accepts exactly the 30 model features and returns the fraud probability, risk level, and recommended action. `GET /` is a health endpoint. The local demo also provides verified individual fraud and legitimate examples at `/demo/fraud` and `/demo/legitimate`.

## Frontend

The plain HTML/CSS/JavaScript dashboard in `frontend/index.html` includes advanced feature entry, verified demo samples, a probability meter, policy guidance, and browser-local demo history. It does not receive the full dataset or present synthetic explainability.

## Local setup and demo

Use a Python environment that contains `fastapi`, `uvicorn`, `joblib`, `pandas`, `scikit-learn`, and `xgboost`.

```powershell
cd C:\Users\ANKAN SEN\Desktop\AI-Risk-Manager
python -m uvicorn api.main:app --reload --port 8000
```

In a second terminal, serve the frontend rather than opening it directly from the filesystem:

```powershell
cd C:\Users\ANKAN SEN\Desktop\AI-Risk-Manager\frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500`, load a verified example, and select **Analyze Transaction**.

## Limitations

This is a prototype fraud-risk decision system using a public benchmark dataset. It has no production authentication, transaction ingestion, model monitoring, or calibrated real-world merchant-loss costs.
