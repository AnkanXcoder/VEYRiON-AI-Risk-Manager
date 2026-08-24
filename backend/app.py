import json
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
# 4. DEMO TRANSACTIONS
#
# These are model-compatible benchmark examples.
# They allow the deployed API to work WITHOUT
# train_test_split.pkl.
# =========================================================

LEGITIMATE_DEMO = {
    "Time": 155400.0,
    "Amount": 0.0,

    "V1": 1.87763102653787,
    "V2": 1.23384772843901,
    "V3": -1.55615714625486,
    "V4": 4.37648657970126,
    "V5": 0.897299673442182,
    "V6": -0.829882783956245,
    "V7": 0.510571445731182,
    "V8": -0.2332812514279,
    "V9": -1.08956046295809,
    "V10": 0.187423368255943,
    "V11": 0.0004703637759045,
    "V12": 0.0352038942118656,
    "V13": 0.664548229327016,
    "V14": -2.94786970023766,
    "V15": -1.29046615706958,
    "V16": 1.14072539363065,
    "V17": 1.85126093600225,
    "V18": 0.205154128243836,
    "V19": -1.91109912141651,
    "V20": -0.193509038180537,
    "V21": -0.181469450784463,
    "V22": -0.286038215429858,
    "V23": 0.158538183274642,
    "V24": 0.104445951640915,
    "V25": 0.0330868308568062,
    "V26": -0.0627383090152664,
    "V27": 0.0101411456085235,
    "V28": 0.0165136506440117
}


REVIEW_DEMO = {
    "Time": 61290.0,
    "Amount": 11.5,

    "V1": 1.2288211502379,
    "V2": -0.0634077165201056,
    "V3": 0.274145142235826,
    "V4": 0.647465021810117,
    "V5": -0.0481345611508765,
    "V6": 0.372073028593297,
    "V7": -0.22423058741343,
    "V8": 0.0799390492455152,
    "V9": 0.640758817066441,
    "V10": -0.273053702248503,
    "V11": -1.25272793883718,
    "V12": 0.465078770741453,
    "V13": 0.400502115321077,
    "V14": -0.292841860600363,
    "V15": -0.10177401599731,
    "V16": -0.399835897844616,
    "V17": 0.0343356567914817,
    "V18": -0.783550254934187,
    "V19": 0.141344900433949,
    "V20": -0.0965659023514416,
    "V21": -0.129554448055005,
    "V22": -0.0837793282428063,
    "V23": -0.151661473916324,
    "V24": -0.700371597289218,
    "V25": 0.598550164523483,
    "V26": 0.491409070563651,
    "V27": 0.0029892597250263,
    "V28": 0.0017822861144491
}


# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================

def validate_transaction(data):
    """
    Make sure all model features are present.
    """

    if not isinstance(data, dict):
        raise ValueError("Transaction must be a JSON object.")

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    return data


def transaction_to_json(transaction):
    """
    Convert transaction values to normal JSON numbers.
    """

    return {
        feature: float(transaction[feature])
        for feature in MODEL_FEATURES
    }


def load_fraud_demo():
    """
    Load the fraud demonstration transaction
    from data/fraud_api_sample.json.
    """

    sample_path = DATA_DIR / "fraud_api_sample.json"

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Fraud demo sample not found: {sample_path}"
        )

    with open(
        sample_path,
        "r",
        encoding="utf-8"
    ) as file:

        sample = json.load(file)

    return validate_transaction(sample)


# =========================================================
# 6. HOME PAGE
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
# 7. HEALTH CHECK
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
# 8. PREDICTION API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "Request body is empty"
            }), 400

        data = validate_transaction(data)

        result = predict_fraud(data)

        return jsonify(result), 200

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:

        return jsonify({

            "error":
                "Prediction failed",

            "details":
                str(error)

        }), 500


# =========================================================
# 9. DEMO FRAUD TRANSACTION
# =========================================================

@app.route("/demo/fraud", methods=["GET"])
def demo_fraud():

    try:

        fraud_sample = load_fraud_demo()

        return jsonify(fraud_sample), 200

    except Exception as error:

        return jsonify({

            "error":
                "Demo fraud sample could not be loaded",

            "details":
                str(error)

        }), 500


# =========================================================
# 10. DEMO LEGITIMATE TRANSACTION
# =========================================================

@app.route("/demo/legitimate", methods=["GET"])
def demo_legitimate():

    try:

        return jsonify(
            LEGITIMATE_DEMO
        ), 200

    except Exception as error:

        return jsonify({

            "error":
                "Demo legitimate sample could not be loaded",

            "details":
                str(error)

        }), 500


# =========================================================
# 11. DEMO REVIEW TRANSACTION
# =========================================================

@app.route("/demo/review", methods=["GET"])
def demo_review():

    try:

        return jsonify(
            REVIEW_DEMO
        ), 200

    except Exception as error:

        return jsonify({

            "error":
                "Demo review sample could not be loaded",

            "details":
                str(error)

        }), 500


# =========================================================
# 12. LEGACY MEDIUM ENDPOINT
# =========================================================

@app.route("/demo/medium", methods=["GET"])
def demo_medium():

    return demo_review()


# =========================================================
# 13. START SERVER
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