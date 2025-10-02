from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import uvicorn
import os

# ------------------------
# Environment Configuration
# ------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_TITLE = os.getenv("API_TITLE", "NASA Weather Prediction API")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# ------------------------
# Load models and features
# ------------------------
rfr = joblib.load("temperature_model.pkl")   # Regressor model
rfc = joblib.load("rain_model.pkl")          # Classifier model

feature_cols_temp = joblib.load("feature_cols_temp.pkl")
feature_cols_rain = joblib.load("feature_cols_rain.pkl")

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="AI-powered weather prediction API for temperature and rainfall forecasting"
)

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "NASA Weather API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

def preprocess_input(date_str, region):
    """
    Preprocess user input (date + region) -> features DataFrame
    """
    # 1) تحويل التاريخ
    date = datetime.strptime(date_str, "%Y-%m-%d")
    month = date.month
    dayofyear = date.timetuple().tm_yday

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    doy_sin = np.sin(2 * np.pi * dayofyear / 365)
    doy_cos = np.cos(2 * np.pi * dayofyear / 365)

    # 2) one-hot للـ region
    region_dict = {"region_North": 0, "region_Middle": 0, "region_South": 0}
    if region in ["North", "Middle", "South"]:
        region_dict[f"region_{region}"] = 1

    # 3) return dict بالـ features المشتركة
    features = {
        "month_sin": month_sin,
        "month_cos": month_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "region_North": region_dict["region_North"],
        "region_Middle": region_dict["region_Middle"],
        "region_South": region_dict["region_South"],
    }
    return features

@app.post("/predict")
def predict(date: str, region: str):
    # --------------------
    # Preprocess input
    # --------------------
    base_features = preprocess_input(date, region)

    # ---- Temperature ----
    row_temp = {c: 0.0 for c in feature_cols_temp}
    row_temp.update(base_features)
    df_temp = pd.DataFrame([row_temp])[feature_cols_temp]
    temp_pred = rfr.predict(df_temp)[0]

    # ---- Rain ----
    row_rain = {c: 0.0 for c in feature_cols_rain}
    row_rain.update(base_features)
    df_rain = pd.DataFrame([row_rain])[feature_cols_rain]
    rain_pred = rfc.predict(df_rain)[0]
    rain_proba = rfc.predict_proba(df_rain)[0][1]  # احتمال المطر

    return {
        "date": date,
        "region": region,
        "predicted_temperature": round(float(temp_pred), 2),
        "rain_status": "Rain" if rain_pred == 1 else "No Rain",
        "rain_probability": round(float(rain_proba), 2)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, debug=DEBUG)
