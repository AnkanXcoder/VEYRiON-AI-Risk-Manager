import json
import joblib
import os

from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    from .predictor import predict_fraud, MODEL_FEATURES
except ImportError:
    from predictor import predict_fraud, MODEL_FEATURES


# =========================================================
# 1. FLASK APPLICATION
# =========================================================

app = Flask(__name__)

CORS(app)


# =========================================================
# 2. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"


# =========================================================
# 3. LOCKED RISK POLICY
# =========================================================

MEDIUM_THRESHOLD = 0.01
HIGH_THRESHOLD = 0.19


# =========================================================
# 4. HOME PAGE
# =========================================================

@app.route("/", methods=["GET"])
def home():

    index_path = FRONTEND_DIR / "index.html"

    if not index_path.exists():

        return jsonify({
            "error": "Frontend index.html not found",
            "details": str(index_path)
        }), 404

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# =========================================================
# 5. HEALTH CHECK
# =========================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "message":
            "VEYRiON AI Risk Manager API is running",

        "status":
            "online",

        "model":
            "XGBoost",

        "policy": {

            "approve_below":
                MEDIUM_THRESHOLD,

            "review_from":
                MEDIUM_THRESHOLD,

            "review_below":
                HIGH_THRESHOLD,

            "block_at_or_above":
                HIGH_THRESHOLD

        }

    }), 200


# =========================================================
# 6. PREDICTION API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify({

                "error":
                    "Request body is empty"

            }), 400


        result = predict_fraud(data)


        return jsonify(
            result
        ), 200


    except ValueError as error:

        return jsonify({

            "error":
                str(error)

        }), 400


    except Exception as error:

        return jsonify({

            "error":
                "Prediction failed",

            "details":
                str(error)

        }), 500


# =========================================================
# 7. LOAD TEST DATA
# =========================================================

def load_test_data():

    split_path = (
        MODELS_DIR
        / "train_test_split.pkl"
    )


    if not split_path.exists():

        raise FileNotFoundError(

            f"Train/test split not found: "
            f"{split_path}"

        )


    split_data = joblib.load(
        split_path
    )


    X_test = split_data["X_test"]

    y_test = split_data["y_test"]


    return X_test, y_test


# =========================================================
# 8. CONVERT TRANSACTION TO JSON
# =========================================================

def transaction_to_json(transaction):

    return {

        feature:
            float(transaction[feature])

        for feature in MODEL_FEATURES

    }


# =========================================================
# 9. LOAD MODEL AND SCALER
# =========================================================

def load_model_and_scaler():

    model_path = (
        MODELS_DIR
        / "xgboost_candidate.pkl"
    )

    scaler_path = (
        MODELS_DIR
        / "scaler.pkl"
    )


    if not model_path.exists():

        raise FileNotFoundError(

            f"XGBoost model not found: "
            f"{model_path}"

        )


    if not scaler_path.exists():

        raise FileNotFoundError(

            f"Scaler not found: "
            f"{scaler_path}"

        )


    model = joblib.load(
        model_path
    )

    scaler = joblib.load(
        scaler_path
    )


    return model, scaler


# =========================================================
# 10. CALCULATE TEST SET PROBABILITIES
# =========================================================

def get_test_probabilities():

    X_test, y_test = load_test_data()

    model, scaler = load_model_and_scaler()


    X_test_ordered = X_test[
        MODEL_FEATURES
    ]


    X_test_scaled = scaler.transform(
        X_test_ordered
    )


    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]


    return (
        X_test,
        y_test,
        probabilities
    )


# =========================================================
# 11. DEMO FRAUD TRANSACTION
# =========================================================

@app.route(
    "/demo/fraud",
    methods=["GET"]
)
def demo_fraud():

    try:

        sample_path = (
            DATA_DIR
            / "fraud_api_sample.json"
        )


        if not sample_path.exists():

            return jsonify({

                "error":
                    "Fraud demo sample not found",

                "details":
                    str(sample_path)

            }), 404


        with open(
            sample_path,
            "r",
            encoding="utf-8"
        ) as file:

            fraud_sample = json.load(
                file
            )


        return jsonify(
            fraud_sample
        ), 200


    except Exception as error:

        return jsonify({

            "error":
                "Demo fraud sample could not be loaded",

            "details":
                str(error)

        }), 500


# =========================================================
# 12. DEMO LEGITIMATE TRANSACTION
# =========================================================

@app.route(
    "/demo/legitimate",
    methods=["GET"]
)
def demo_legitimate():

    try:

        (
            X_test,
            y_test,
            probabilities
        ) = get_test_probabilities()


        if hasattr(
            y_test,
            "to_numpy"
        ):

            y_values = y_test.to_numpy()

        else:

            y_values = y_test


        legitimate_mask = (

            (y_values == 0)

            &

            (
                probabilities
                < MEDIUM_THRESHOLD
            )

        )


        matching_positions = [

            i

            for i, matched
            in enumerate(
                legitimate_mask
            )

            if matched

        ]


        if not matching_positions:

            return jsonify({

                "error":
                    "No low-risk legitimate "
                    "transaction found."

            }), 404


        position = matching_positions[0]


        transaction_index = (
            X_test.index[position]
        )


        transaction = X_test.loc[
            transaction_index,
            MODEL_FEATURES
        ]


        return jsonify(
            transaction_to_json(
                transaction
            )
        ), 200


    except Exception as error:

        return jsonify({

            "error":
                "Demo legitimate sample "
                "could not be loaded",

            "details":
                str(error)

        }), 500


# =========================================================
# 13. DEMO REVIEW TRANSACTION
# =========================================================

@app.route(
    "/demo/review",
    methods=["GET"]
)
def demo_review():

    try:

        (
            X_test,
            y_test,
            probabilities
        ) = get_test_probabilities()


        if hasattr(
            y_test,
            "to_numpy"
        ):

            y_values = y_test.to_numpy()

        else:

            y_values = y_test


        # -------------------------------------------------
        # Prefer legitimate transaction in REVIEW range
        # -------------------------------------------------

        review_mask = (

            (y_values == 0)

            &

            (
                probabilities
                >= MEDIUM_THRESHOLD
            )

            &

            (
                probabilities
                < HIGH_THRESHOLD
            )

        )


        matching_positions = [

            i

            for i, matched
            in enumerate(
                review_mask
            )

            if matched

        ]


        # -------------------------------------------------
        # Fallback: any transaction in REVIEW range
        # -------------------------------------------------

        if not matching_positions:

            review_mask = (

                (
                    probabilities
                    >= MEDIUM_THRESHOLD
                )

                &

                (
                    probabilities
                    < HIGH_THRESHOLD
                )

            )


            matching_positions = [

                i

                for i, matched
                in enumerate(
                    review_mask
                )

                if matched

            ]


        if not matching_positions:

            return jsonify({

                "error":
                    "No real review-range "
                    "transaction found.",

                "details":
                    "No transaction in the test "
                    "set has probability between "
                    "1% and 19%."

            }), 404


        # -------------------------------------------------
        # Select transaction closest to 10%
        # -------------------------------------------------

        target_probability = 0.10


        position = min(

            matching_positions,

            key=lambda i:

                abs(
                    probabilities[i]
                    - target_probability
                )

        )


        transaction_index = (
            X_test.index[position]
        )


        transaction = X_test.loc[
            transaction_index,
            MODEL_FEATURES
        ]


        return jsonify(
            transaction_to_json(
                transaction
            )
        ), 200


    except Exception as error:

        return jsonify({

            "error":
                "Demo review sample "
                "could not be loaded",

            "details":
                str(error)

        }), 500


# =========================================================
# 14. LEGACY MEDIUM ENDPOINT
# =========================================================

@app.route(
    "/demo/medium",
    methods=["GET"]
)
def demo_medium():

    return demo_review()


# =========================================================
# 15. START SERVER
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("VEYRiON AI RISK MANAGER")
    print("=" * 60)

    print("Model       : XGBoost")
    print("Approve     : < 1%")
    print("Review      : 1% - < 19%")
    print("Block       : >= 19%")
    print("Health      : /api/health")

    print("=" * 60)
    print()


    app.run(

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),

        debug=False

    )