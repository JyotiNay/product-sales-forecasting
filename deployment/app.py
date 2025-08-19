
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model at startup
MODEL_PATH = os.getenv("MODEL_PATH", "ml_outputs/best_model_pipeline.joblib")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict_features", methods=["POST"])
def predict_features():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    features = data.get("features")

    if not features:
        return jsonify({"error": "No features provided"}), 400

    X = pd.DataFrame([features])
    pred = model.predict(X)[0]

    return jsonify({"prediction": float(pred)})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
