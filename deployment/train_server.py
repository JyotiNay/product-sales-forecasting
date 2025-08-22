# deployment/train_server.py
from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

def synthesize_data(n=50000, seed=42):
    rng = np.random.default_rng(seed)
    categories = np.array(["A","B","C","D"])
    price = rng.uniform(1, 100, size=n).round(2)
    is_promo = rng.binomial(1, 0.30, size=n).astype(int)
    stock = rng.integers(0, 300, size=n)
    category = rng.choice(categories, size=n, p=[0.25, 0.35, 0.25, 0.15])

    # simple seasonal-ish effect via synthetic months
    month = rng.integers(1, 13, size=n)
    cat_effect = pd.Series(category).map({"A":5,"B":12,"C":0,"D":-5}).values
    seasonal = np.where(np.isin(month,[11,12]), 10, 0)
    stock_effect = np.where(stock > 50, 5, 0)

    mu = (20 + (100 - price)*0.5 + is_promo*(12 + 0.1*(100 - price))
          + stock_effect + cat_effect + seasonal)
    noise = rng.normal(0, 5, size=n)
    sales = np.maximum(0, np.round(mu + noise)).astype(int)

    return pd.DataFrame({
        "price": price,
        "is_promo": is_promo,
        "stock": stock,
        "category": category,
        "sales": sales
    })

def main():
    # If you later add a real CSV, replace this with a loader.
    df = synthesize_data()

    X = df.drop(columns=["sales"])
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric = ["price","stock","is_promo"]
    categorical = ["category"]

    pre = ColumnTransformer([
        ("num","passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    model = GradientBoostingRegressor(random_state=42)
    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    mape = float(np.mean(np.abs((y_test - preds) / np.clip(y_test, 1, None))))

    joblib.dump(pipe, ART_DIR / "model.joblib")
    with open(ART_DIR / "metrics.json","w") as f:
        json.dump({"rmse": rmse, "mae": mae, "mape": mape}, f)

    print("[ok] Trained and saved to artifacts/model.joblib with metrics:", rmse, mae, mape)

if __name__ == "__main__":
    main()

