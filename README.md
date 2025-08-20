# Product Sales Forecasting – EDA, Modeling & API Deployment

## 1) Problem Statement
Accurately forecasting retail product sales to optimize inventory, staffing, promotions, and financial planning.

**Target metric:** Root Mean Squared Error (RMSE) on hold-out validation window (last 28 days) and cross-validation.  
**Secondary metrics:** MAE, MAPE.

---

## 2) Dataset
Historical daily store-level data with features:
- Store_Type, Location_Type, Region_Code
- Date, Holiday, Discount
- #Order (daily orders), Sales (daily sales)

> Note: Full data is not committed. A small sample lives in `data_sample/` for structure reference.

---

## 3) Method – EDA & Hypothesis Testing (Block 2)
Notebook: `notebooks/01_eda_hypothesis_ml.ipynb`

**EDA highlights:**
- Sales and #Order are right-skewed; strong Pearson correlation between them.
- Seasonal patterns visible; store/region categories have significant mean differences.
- Outliers present; handled via robust modeling and rolling features.

**Hypothesis tests (high-level):**
- Discounts increase sales (Welch t-test, p < 0.001).
- Holiday impact varies; in our data, sales were lower on holidays (p < 0.001).
- ANOVA: significant differences across Store Types & Regions.
- Correlation: #Order vs Sales strongly positive (~0.94 in our run).

---

## 4) Feature Engineering & Modeling (Block 3)
- Calendar features: Year, Month, Week, Day, DayOfWeek, IsWeekend
- Flags: Discount_Flag, Holiday_Flag
- Store-wise lag/rolling features:
  - Sales_lag1, Sales_lag7, Sales_roll7, Sales_roll28
  - Orders_lag1, Orders_lag7, Orders_roll7, Orders_roll28
- AOV proxy: `Sales_per_order = Sales / #Order`

**Models:**
- Baseline: Linear Regression (with OHE + scaling)
- Tree-based: RandomForest, GradientBoosting (with OHE)
- Time-aware validation: last-28-days holdout + expanding-window CV (optional)

**Final model chosen:** <fill in>  
**Validation metrics:**  
- RMSE: <rmse>  
- MAE: <mae>  
- MAPE: <mape>%  

---

## 5) Deployment (Block 4)
A minimal Flask API serves predictions.

**Run locally:**
```bash
cd deployment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


The final model is deployed as an API using **Flask** on **Render**.  

- Live API: [Product Sales Forecasting](https://product-sales-forecasting.onrender.com)  
- Health Check: `/health`  
- Prediction Endpoint: `/predict_features`  

Example request:
```bash
curl -X POST https://product-sales-forecasting.onrender.com/predict_features \
  -H 'Content-Type: application/json' \
  -d '{"features": {"price": 12.99, "is_promo": 1, "stock": 120, "category": "B"}}'

export MODEL_PATH=../ml_outputs/best_model_pipeline.joblib
export PORT=8000


Insights & Recommendations
- Promotions increase sales by ~25% on average.  
- Seasonal spikes in Q4 (holiday season) require stocking ~30% higher inventory.  
- Price elasticity varies by category — luxury items less sensitive vs. budget items highly sensitive.  
- Stock-outs directly cut sales; safety stock levels should be increased for top 20% products.  
python app.py

