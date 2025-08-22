# deployment/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import joblib, json, traceback, subprocess, sys

# --- Create app FIRST (before any decorators) ---
app = Flask(__name__)
CORS(app)

# --- Artifacts & model loading ---
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)
MODEL_PATH = ART_DIR / "model.joblib"
METRICS_PATH = ART_DIR / "metrics.json"

def ensure_model():
    """Ensure a trained model exists; if not, run deployment/train_server.py."""
    if not MODEL_PATH.exists():
        try:
            import subprocess, sys
            print("[boot] model missing → training via deployment/train_server.py")
            subprocess.check_call([sys.executable, "deployment/train_server.py"])
        except Exception as e:
            print("[warn] Training failed:", e)

ensure_model()

# Load pipeline
pipe = None
if MODEL_PATH.exists():
    pipe = joblib.load(MODEL_PATH)

metrics = {}
if METRICS_PATH.exists():
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}

# --- Helpers ---
def _coerce(rec: dict) -> dict:
    req = ["price", "is_promo", "stock", "category"]
    missing = [k for k in req if k not in rec]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    def to_float(x):
        try:
            return float(x)
        except Exception:
            raise ValueError(f"Cannot parse numeric value for price: {x}")

    def to_int_boolish(x):
        if isinstance(x, str):
            t = x.strip().lower()
            if t in {"1","true","t","yes","y"}: return 1
            if t in {"0","false","f","no","n"}: return 0
        return int(x)

    def to_int(x):
        try:
            return int(x)
        except Exception:
            raise ValueError(f"Cannot parse integer value: {x}")

    return {
        "price": to_float(rec["price"]),
        "is_promo": to_int_boolish(rec["is_promo"]),
        "stock": to_int(rec["stock"]),
        "category": str(rec["category"]),
    }

# --- Routes ---
@app.get("/")
def index():
    return (
        "<h2>Product Sales Forecasting API</h2>"
        "<p>Endpoints available:</p>"
        "<ul>"
        "<li><a href='/health'>/health</a> → check if service & model are ok</li>"
        "<li>POST /predict_features → send JSON with flat body, 'features', or 'instances'</li>"
        "</ul>"
        "<p>Example request:</p>"
        "<pre>curl -X POST https://product-sales-forecasting.onrender.com/predict_features "
        "-H 'Content-Type: application/json' "
        "-d '{\"features\": {\"price\": 12.99, \"is_promo\": 1, \"stock\": 120, \"category\": \"B\"}}'</pre>"
    )

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": pipe is not None,
        "metrics": metrics
    })

@app.post("/predict_features")
def predict_features():
    if pipe is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if not payload:
        return jsonify({"error": "Empty payload"}), 400

    try:
        # Accept flat JSON
        if isinstance(payload, dict) and all(k in payload for k in ["price","is_promo","stock","category"]):
            X = pd.DataFrame([_coerce(payload)])
            yhat = float(pipe.predict(X)[0])
            return jsonify({"prediction": yhat})

        # Accept {"features": {...}}
        if "features" in payload:
            X = pd.DataFrame([_coerce(payload["features"])])
            yhat = float(pipe.predict(X)[0])
            return jsonify({"prediction": yhat})

        # Accept {"instances": [{...}, {...}]}
        if "instances" in payload and isinstance(payload["instances"], list):
            rows = [_coerce(r) for r in payload["instances"]]
            X = pd.DataFrame(rows)
            preds = [float(v) for v in pipe.predict(X)]
            return jsonify({"predictions": preds, "count": len(preds)})

        return jsonify({"error": "Send a flat JSON, or {'features': {...}}, or {'instances': [...] }"}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400
