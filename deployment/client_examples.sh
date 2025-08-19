#!/usr/bin/env bash

# Health
curl http://127.0.0.1:8000/health

# Predict with engineered features
curl -X POST http://127.0.0.1:8000/predict_features   -H "Content-Type: application/json"   -d '{
    "features": {
      "Store_Type":"S1","Location_Type":"L1","Region_Code":"R1","Discount":"Yes",
      "Discount_Flag":1,"Holiday_Flag":0,"IsWeekend":0,
      "#Order":120,"Year":2019,"Month":1,"Week":5,"Day":10,"DayOfWeek":3,
      "Sales_lag1":15250.0,"Sales_lag7":14880.0,"Sales_roll7":15020.0,"Sales_roll28":14700.0,
      "Orders_lag1":118,"Orders_lag7":113,"Orders_roll7":116.0,"Orders_roll28":111.0,
      "Sales_per_order":125.5
    }
  }'
