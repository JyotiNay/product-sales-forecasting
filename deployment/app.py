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

# 👇 NEW: Home route so visiting "/" works
@app.route("/", methods=["GET"])
def home():
    return (
        "<h2>Product Sales Forecasting API</h2>"
        "<p>Endpoints available:</p>"
        "<ul>"
        "<li><a href='/health'>/health</a> → check if service & model are ok</li>"
        "<li>POST /predict_features → send JSON with 'features'</li>"
        "</ul>"
        "<p>Example request:</p>"
        "<pre>curl -X POST https://product-sales-forecasting.onrender.com/predict_features "
        "-H 'Content-Type: application/json' "
        "-d '{\"features\": {\"price\": 12.99, \"is_promo\": 1, \"stock\": 120, \"category\": \"B\"}}'</pre>"
    ), 200

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
