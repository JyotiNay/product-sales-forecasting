# Product Sales Forecasting – EDA, Modeling & API Deployment

1) Problem Statement
Accurately forecasting retail product sales to optimize inventory, staffing, promotions, and financial planning.
Target metric: Root Mean Squared Error (RMSE) on a time-ordered hold‑out validation window (last 28 days) and cross‑validation.
Secondary metrics: MAE, MAPE.
Final validation performance (Random Forest):
RMSE: 561.51
MAE: 226.87
MAPE: 45.0%
2) Dataset
Historical daily store‑level data with features:
Store_Type, Location_Type, Region_Code
Date, Holiday, Discount
#Order (daily orders), Sales (daily sales)
Full data is not committed. A small sample lives in data_sample/ for structure reference.
3) Method – EDA & Hypothesis Testing
Notebook: notebooks/01_eda_hypothesis_ml.ipynb
EDA highlights (examples):
Sales and #Order are right‑skewed; strong Pearson correlation between them.
Clear seasonality; store/region categories show significant mean differences.
Outliers handled via robust modeling and rolling features.
Hypothesis tests (illustrations):
Discounts increase sales (Welch t‑test, p < 0.001).
Holiday impact varies; in our data, sales were lower on holidays (p < 0.001).
ANOVA confirms significant differences across Store Types & Regions.
Correlation: #Order vs Sales ~0.94 in our run.
4) Feature Engineering & Modeling
Calendar features: Year, Month, Week, Day, DayOfWeek, IsWeekend
Flags: Discount_Flag, Holiday_Flag
Store‑wise lag/rolling features:
Sales_lag1, Sales_lag7, Sales_roll7, Sales_roll28
Orders_lag1, Orders_lag7, Orders_roll7, Orders_roll28
AOV proxy: Sales_per_order = Sales / #Order
Models evaluated:
Baseline: Linear Regression (OHE + scaling)
Tree‑based: RandomForest, GradientBoosting (with OHE)
Time‑aware validation: last‑28‑days holdout + (optional) expanding‑window CV
Final model chosen: RandomForest Regressor
Validation metrics: RMSE 561.51, MAE 226.87, MAPE 45.0%
5) Deployment
A minimal Flask API serves predictions (deployed on Render).
Live API: https://product-sales-forecasting.onrender.com
Health Check: /health
Prediction Endpoint: /predict_features
Run locally
cd deployment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
Example request
curl -X POST https://product-sales-forecasting.onrender.com/predict_features \
  -H 'Content-Type: application/json' \
  -d '{"features": {"price": 12.99, "is_promo": 1, "stock": 120, "category": "B"}}'
Example response (excerpt)
{"prediction": 47469.30}
6) Insights & Recommendations
Promotions increase sales on average; size and duration matter—align with top Store_Type/Region_Code.
Seasonal spikes (e.g., late Q4) require proactive inventory + staffing buffers.
Price elasticity varies by category—luxury items less sensitive vs. budget items highly sensitive.
Prevent stock‑outs: increase safety stock for the top 20% of products by revenue/velocity.
7) Tableau Dashboards
Public link: https://public.tableau.com/app/profile/jyotirmayee.nayak/viz/ProductSalesForecasting_17558078988380/ForecastEvaluation
Sales Performance
Regional Sales Analysis
Promotional Impact
Operational Insights
Forecast Evaluation
8) Reproducibility
Create env & install deps.
Run notebooks/01_eda_hypothesis_ml.ipynb to regenerate features and model.
Export Tableau CSVs into /tableau.
Open dashboards via Tableau Public link.
## Live API
Base URL: https://product-sales-forecasting.onrender.com

### Health
`GET /health`
```bash
curl https://product-sales-forecasting.onrender.com/health
Predict (single)
POST /predict_features
curl -X POST https://product-sales-forecasting.onrender.com/predict_features \
  -H "Content-Type: application/json" \
  -d '{"features":{"price":12.99,"is_promo":1,"stock":120,"category":"B"}}'
Predict (batch)
curl -X POST https://product-sales-forecasting.onrender.com/predict_features \
  -H "Content-Type: application/json" \
  -d '{"instances":[
        {"price":12.99,"is_promo":1,"stock":120,"category":"B"},
        {"price":49.99,"is_promo":0,"stock":30,"category":"D"}
      ]}'

---

If you want, I can also:
- generate a **Postman collection JSON** for import, or
- write a **1-page Medium-ready “deployment appendix”** with these exact commands and screenshots checklist.

What do you want to tackle next: Postman collection, README polish, or final Loom script with your real URL stitched in?
10) Repository
.
├── notebooks/EDA_and_Modeling.ipynb
├── deployment/app.py
├── deployment/requirements.txt
├── ml_outputs/best_model_pipeline.joblib
├── tableau/*.csv
├── screenshots/*.png
└── README.md


