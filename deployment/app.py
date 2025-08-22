from flask import request, jsonify
import pandas as pd
import traceback

def _coerce(rec: dict) -> dict:
    req = ["price", "is_promo", "stock", "category"]
    missing = [k for k in req if k not in rec]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # permissive coercion
    def to_float(x):
        try:
            return float(x)
        except Exception:
            raise ValueError(f"Cannot parse numeric value for price: {x}")

    def to_int_boolish(x):
        # allow 1/0, true/false strings
        if isinstance(x, str):
            t = x.strip().lower()
            if t in {"1","true","t","yes","y"}: return 1
            if t in {"0","false","f","no","n"}: return 0
            # fall-through to int cast below
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

@app.post("/predict_features")
def predict_features():
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if not payload:
        return jsonify({"error": "Empty payload"}), 400

    try:
        # Accept flat body
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

        return jsonify({"error": "Send either a flat JSON, or {'features': {...}}, or {'instances': [...] }"}), 400

    except Exception as e:
        # print full traceback to Render logs (so 500s become debuggable)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400
