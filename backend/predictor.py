import joblib
import pandas as pd
import shap

from pathlib import Path


# =========================================================
# 1. PROJECT ROOT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# =========================================================
# 2. MODEL PATHS
# =========================================================

MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "xgboost_candidate.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
RISK_POLICY_PATH = MODELS_DIR / "risk_policy.pkl"


# =========================================================
# 3. LOAD SAVED ARTIFACTS
# =========================================================

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

risk_policy = joblib.load(RISK_POLICY_PATH)


# =========================================================
# 4. SHAP EXPLAINER
# =========================================================

explainer = shap.TreeExplainer(model)


# =========================================================
# 5. MODEL FEATURES
# =========================================================

MODEL_FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]


# =========================================================
# 6. INPUT VALIDATION
# =========================================================

def validate_transaction(transaction_features):

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in transaction_features
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )


    # Check numeric values

    for feature in MODEL_FEATURES:

        value = transaction_features[feature]

        if not isinstance(value, (int, float)):

            raise ValueError(
                f"{feature} must be numeric."
            )


    # Amount cannot be negative

    if transaction_features["Amount"] < 0:

        raise ValueError(
            "Amount cannot be negative."
        )


# =========================================================
# 7. RISK DECISION
# =========================================================

def risk_decision(fraud_probability):

    threshold_high = float(
        risk_policy["threshold_high"]
    )

    threshold_medium = float(
        risk_policy["threshold_medium"]
    )


    if fraud_probability >= threshold_high:

        return "HIGH", "BLOCK"


    elif fraud_probability >= threshold_medium:

        return "MEDIUM", "REVIEW"


    else:

        return "LOW", "APPROVE"


# =========================================================
# 8. FRAUD PREDICTION
# =========================================================

def predict_fraud(transaction_features):

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    validate_transaction(
        transaction_features
    )


    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

    input_df = pd.DataFrame(
        [
            [
                transaction_features[feature]
                for feature in MODEL_FEATURES
            ]
        ],
        columns=MODEL_FEATURES
    )


    # -----------------------------------------------------
    # Scale features
    # -----------------------------------------------------

    input_scaled = scaler.transform(
        input_df
    )


    # -----------------------------------------------------
    # Predict fraud probability
    # -----------------------------------------------------

    fraud_probability = model.predict_proba(
        input_scaled
    )[0, 1]


    # -----------------------------------------------------
    # Apply saved risk policy
    # -----------------------------------------------------

    risk_level, action = risk_decision(
        fraud_probability
    )


    # -----------------------------------------------------
    # SHAP explanation
    # -----------------------------------------------------

    shap_explanation = explainer(
        input_scaled
    )


    shap_values = shap_explanation.values[0]


    # -----------------------------------------------------
    # Create SHAP importance table
    # -----------------------------------------------------

    shap_importance = pd.DataFrame({

        "feature": MODEL_FEATURES,

        "shap_value": shap_values

    })


    shap_importance["absolute_shap"] = (
        shap_importance["shap_value"].abs()
    )


    # Strongest contributors first

    shap_importance = shap_importance.sort_values(
        "absolute_shap",
        ascending=False
    )


    # -----------------------------------------------------
    # Top 5 contributors
    # -----------------------------------------------------

    top_shap_features = (

        shap_importance

        .head(5)

        [["feature", "shap_value"]]

        .to_dict(
            orient="records"
        )

    )


    # -----------------------------------------------------
    # Final API response
    # -----------------------------------------------------

    return {

        "fraud_probability": round(
            float(fraud_probability),
            4
        ),

        "risk_level": risk_level,

        "action": action,

        "top_shap_features": [

            {
                "feature": item["feature"],

                "shap_value": round(
                    float(item["shap_value"]),
                    4
                )

            }

            for item in top_shap_features

        ]

    }