import os
import json
import boto3
import tempfile
from math import exp

# -----------------------------
# ENVIRONMENT VARIABLES
# -----------------------------
S3_BUCKET = os.environ.get("MODEL_S3_BUCKET")
S3_KEY = os.environ.get("MODEL_S3_KEY", "model/model_params.json")

# -----------------------------
# GLOBALS
# -----------------------------
_smodel = None
s3 = boto3.client('s3')

# -----------------------------
# LOAD MODEL FROM S3
# -----------------------------
def download_model_json(bucket, key, tmp_path):
    s3.download_file(bucket, key, tmp_path)

def load_model():
    global _smodel
    if _smodel is not None:
        return _smodel
    if not S3_BUCKET:
        raise RuntimeError("MODEL_S3_BUCKET environment variable not set")
    tmpfile = os.path.join(tempfile.gettempdir(), "model_params.json")
    download_model_json(S3_BUCKET, S3_KEY, tmpfile)
    with open(tmpfile, "r") as f:
        _smodel = json.load(f)
    return _smodel

# -----------------------------
# MODEL UTILITIES
# -----------------------------
def sigmoid(x):
    return 1 / (1 + exp(-x))

def scale_features(features, mean, scale):
    return [(f - m) / s if s != 0 else 0.0 for f, m, s in zip(features, mean, scale)]

def validate_features(features):
    if not isinstance(features, (list, tuple)):
        raise ValueError("features must be a list")
    if len(features) != 6:
        raise ValueError("features must have length 6 (age, time, labs, meds, diagnoses, insulin)")
    return [float(x) for x in features]

# -----------------------------
# MAIN LAMBDA HANDLER
# -----------------------------
def lambda_handler(event, context):
    try:
        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body)

        if not body:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing body"})
            }

        # -------------------------
        # Accept CUSTOM JSON INPUTS
        # -------------------------
        if "features" not in body:
            required_fields = [
                "age",
                "time_in_hospital",
                "num_lab_procedures",
                "num_medications",
                "number_diagnoses",
                "insulin"
            ]

            # Validate all fields exist
            for f in required_fields:
                if f not in body:
                    return {
                        "statusCode": 400,
                        "body": json.dumps({"error": f"Missing field '{f}'"})
                    }

            # Convert to list expected by model
            features = [
                body["age"],
                body["time_in_hospital"],
                body["num_lab_procedures"],
                body["num_medications"],
                body["number_diagnoses"],
                body["insulin"]
            ]
        else:
            # If user sends "features" list directly
            features = body["features"]

        # Validate & preprocess
        X = validate_features(features)

        # Load model
        model = load_model()
        mean = model["scaler_mean"]
        scale = model["scaler_scale"]
        coef = model["coef"]
        intercept = model["intercept"]

        # Predict
        Xs = scale_features(X, mean, scale)
        logit = intercept + sum(c * x for c, x in zip(coef, Xs))
        proba = sigmoid(logit)

        prediction = 1 if proba >= 0.5 else 0

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "prediction": prediction,
                "probability": float(proba)
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

