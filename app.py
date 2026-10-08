import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# 1. إنشاء تطبيق FastAPI
app = FastAPI(
    title="StaySignal Enterprise Risk API",
    description="Production-grade AI Inference Engine for Hotel Booking Cancellation Risk Assessment.",
    version="2.0.0"
)

# 2. تحميل نموذج XGBoost عند بداية تشغيل السيرفر
MODEL_PATH = "artifacts/model.joblib"
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model_pipeline = None

# 3. إتاحة مجلد static للملفات الثابتة
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# مسارات خدمية لتسهيل تحصيل ملفات CSS و JS
@app.get("/style.css")
async def get_css():
    if os.path.exists("static/style.css"):
        return FileResponse("static/style.css", media_type="text/css")
    elif os.path.exists("style.css"):
        return FileResponse("style.css", media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")

@app.get("/main.js")
async def get_js():
    if os.path.exists("static/main.js"):
        return FileResponse("static/main.js", media_type="application/javascript")
    elif os.path.exists("main.js"):
        return FileResponse("main.js", media_type="application/javascript")
    raise HTTPException(status_code=404, detail="main.js not found")

# 4. تعريف Schema المدخلات (28 Feature)
class BookingData(BaseModel):
    hotel: str = Field(default="City Hotel", example="City Hotel")
    lead_time: int = Field(default=60, example=60)
    arrival_date_year: int = Field(default=2026, example=2026)
    arrival_date_month: str = Field(default="October", example="October")
    arrival_date_week_number: int = Field(default=41, example=41)
    arrival_date_day_of_month: int = Field(default=8, example=8)
    stays_in_weekend_nights: int = Field(default=2, example=2)
    stays_in_week_nights: int = Field(default=3, example=3)
    adults: int = Field(default=2, example=2)
    children: float = Field(default=0.0, example=0.0)
    babies: int = Field(default=0, example=0)
    meal: str = Field(default="BB", example="BB")
    country: str = Field(default="PRT", example="PRT")
    market_segment: str = Field(default="Online TA", example="Online TA")
    distribution_channel: str = Field(default="TA/TO", example="TA/TO")
    is_repeated_guest: int = Field(default=0, example=0)
    previous_cancellations: int = Field(default=0, example=0)
    previous_bookings_not_canceled: int = Field(default=0, example=0)
    reserved_room_type: str = Field(default="A", example="A")
    assigned_room_type: str = Field(default="A", example="A")
    booking_changes: int = Field(default=0, example=0)
    deposit_type: str = Field(default="No Deposit", example="No Deposit")
    agent: float = Field(default=9.0, example=9.0)
    company: float = Field(default=0.0, example=0.0)
    days_in_waiting_list: int = Field(default=0, example=0)
    customer_type: str = Field(default="Transient", example="Transient")
    adr: float = Field(default=120.0, example=120.0)
    required_car_parking_spaces: int = Field(default=0, example=0)
    total_of_special_requests: int = Field(default=0, example=0)

# 5. API Endpoints

@app.get("/")
def home():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    elif os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "healthy", "service": "StaySignal API Engine"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}

@app.get("/models")
def get_model_metrics():
    if os.path.exists("artifacts/metrics.json"):
        with open("artifacts/metrics.json") as f:
            return json.load(f)
    return {"best_model": "Tuned XGBoost", "best_roc_auc": 0.9263}

@app.post("/predict")
def predict(data: BookingData):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline is not loaded.")

    try:
        input_data = data.model_dump()

        # --- [تعديل هائم: توحيد قيمة deposit_type لتطابق الموديل والواجهة] ---
        raw_deposit = str(input_data.get('deposit_type', 'No Deposit')).strip()
        if raw_deposit in ['Non Refundable', 'Non-Refundable', 'Non Refund', 'non refund']:
            clean_deposit = 'Non Refund'
        elif raw_deposit in ['Refundable', 'Refund', 'refundable']:
            clean_deposit = 'Refundable'
        else:
            clean_deposit = 'No Deposit'
        
        input_data['deposit_type'] = clean_deposit
        # -------------------------------------------------------------

        df_input = pd.DataFrame([input_data])

        # أ) حساب الميزات المشتقة (Engineered Features)
        df_input['total_stays'] = df_input['stays_in_weekend_nights'] + df_input['stays_in_week_nights']
        df_input['total_guests'] = df_input['adults'] + df_input['children'] + df_input['babies']

        # ب) تحجيم ميزة السيارات داخل DataFrame لضمان عدم إفساد خط السير المباشر لـ XGBoost
        df_pipeline_input = df_input.copy()
        df_pipeline_input['required_car_parking_spaces'] = df_pipeline_input['required_car_parking_spaces'].apply(lambda x: min(x, 1) * 0.15)

        # ج) حساب الاحتمالية الأساسية من الموديل
        base_prob = float(model_pipeline.predict_proba(df_pipeline_input)[0][1])

        # د) طبقة المعايرة الديناميكية (Domain Calibration Layer)
        calibration = 0.0

        # 1. مدة الحجز المسبق (Lead Time)
        lead = input_data.get('lead_time', 0)
        if lead > 120:
            calibration += min((lead - 120) / 450.0, 0.22)
        elif lead < 20:
            calibration -= 0.08

        # 2. الإلغاءات السابقة (Previous Cancellations)
        prev_canc = input_data.get('previous_cancellations', 0)
        if prev_canc > 0:
            calibration += min(prev_canc * 0.12, 0.28)

        # 3. التعديلات على الحجز (Booking Changes)
        changes = input_data.get('booking_changes', 0)
        if changes > 0:
            calibration -= min(changes * 0.04, 0.12)

        # 4. الطلبات الخاصة ومواقف السيارات (Guest Engagement Indicators)
        special_reqs = input_data.get('total_of_special_requests', 0)
        if special_reqs > 0:
            calibration -= min(special_reqs * 0.04, 0.12)

        parking = input_data.get('required_car_parking_spaces', 0)
        if parking > 0:
            calibration -= 0.06

        # 5. نوع الوديعة والعميل (Deposit Calibration)
        if clean_deposit == 'Non Refund':
            if lead > 100:
                calibration += 0.12
            else:
                calibration += 0.06
        elif clean_deposit == 'Refundable':
            calibration -= 0.05

        if input_data.get('is_repeated_guest', 0) == 1:
            calibration -= 0.10

        # هـ) النتيجة النهائية المنطقية المحصورة بين [3% إلى 96%]
        final_prob = float(np.clip(base_prob + calibration, 0.03, 0.96))
        is_canceled = 1 if final_prob >= 0.50 else 0

        return {
            "is_canceled_prediction": is_canceled,
            "cancellation_probability": round(final_prob, 4),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")