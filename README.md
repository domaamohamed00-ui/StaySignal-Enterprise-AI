<div align="center">

# 🏨 StaySignal Enterprise AI

### Production-Grade MLOps Platform for Real-Time Hotel Booking Cancellation Risk Intelligence

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-EC6B23)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Railway](https://img.shields.io/badge/Deployed%20on-Railway-0B0D0E?logo=railway&logoColor=white)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9263-success)

</div>

---

## 📌 Overview

**StaySignal** predicts, at booking time, how likely a hotel reservation is to be **cancelled**. It turns a trained XGBoost model into a real-time, containerised inference service with a web dashboard, so revenue and operations teams can flag high-risk bookings and act early (overbooking strategy, deposit policy, targeted outreach).

The project covers the full ML lifecycle: data → EDA → training → versioned artifacts → API → Docker → GitHub → cloud deployment.

> Dataset: [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (`hotel_bookings.csv`, Kaggle).

## 🏗️ Architecture

![StaySignal Architecture](architecture-advanced.png)

| Layer | What happens |
|---|---|
| **Offline training** | Raw data → EDA & preprocessing → tuned XGBoost training → exported to `artifacts/` (`model.joblib`, `metrics.json`) |
| **Inference service** | FastAPI loads the model once at startup, validates input with Pydantic, engineers features, predicts, then applies a domain calibration layer |
| **Dashboard** | Vanilla-JS single-page UI (`static/`) served directly by FastAPI |
| **Delivery** | Dockerfile → GitHub → Railway, exposed as a public web app + REST API |

## ✨ Key Features

- **Real-time risk scoring** – cancellation probability and a binary prediction per booking.
- **Full preprocessing + model in one pipeline** (`joblib`) – the API receives raw booking fields, no manual encoding.
- **Strict input contract** – 29-field `BookingData` Pydantic schema with sensible defaults and auto-generated Swagger docs.
- **Engineered features** – `total_stays` (weekend + week nights) and `total_guests` (adults + children + babies).
- **Domain calibration layer** – business-rule adjustments on top of the model output (see below).
- **Health & model-metrics endpoints** for monitoring and dashboards.
- **Container-ready** – one `Dockerfile`, honours the `PORT` env var (Railway/Heroku style).

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the dashboard (`static/index.html`), or a health JSON if absent |
| `GET` | `/health` | Service status + whether the model is loaded |
| `GET` | `/models` | Model metrics from `artifacts/metrics.json` |
| `POST` | `/predict` | Cancellation risk for one booking |
| `GET` | `/docs` | Interactive Swagger UI |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hotel": "City Hotel",
    "lead_time": 180,
    "arrival_date_year": 2026,
    "arrival_date_month": "October",
    "arrival_date_week_number": 41,
    "arrival_date_day_of_month": 8,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 3,
    "adults": 2,
    "market_segment": "Online TA",
    "deposit_type": "No Deposit",
    "previous_cancellations": 1,
    "adr": 120.0,
    "total_of_special_requests": 0
  }'
```

```json
{
  "is_canceled_prediction": 1,
  "cancellation_probability": 0.7312,
  "status": "success"
}
```

> All fields have defaults, so partial payloads are accepted. Values above are illustrative.

## 🧠 Prediction Logic

1. **Validate** the payload against the `BookingData` schema.
2. **Normalise** `deposit_type` (`Non Refundable` / `Refundable` / `No Deposit`) to the values the model was trained on.
3. **Engineer** `total_stays` and `total_guests`.
4. **Score** with the XGBoost pipeline → base probability via `predict_proba`.
5. **Calibrate** with transparent business rules:

   | Signal | Effect on risk |
   |---|---|
   | Very long lead time (> 120 days) | ↑ (capped) |
   | Very short lead time (< 20 days) | ↓ |
   | Previous cancellations | ↑ (capped) |
   | Booking changes, special requests, parking request | ↓ (capped) |
   | Non-refundable deposit | ↑ (more if lead time > 100) |
   | Refundable deposit / repeated guest | ↓ |

6. **Decide** – final probability is clipped to **[3 %, 96 %]**; a booking is flagged as *cancelled* when probability ≥ **0.50**.

## 📊 Model Performance

| Model | ROC-AUC |
|---|---|
| Tuned XGBoost (best) | **0.9263** |

Full metrics are served live at `GET /models`.

## 📁 Project Structure

```
StaySignal-Enterprise-AI/
├── app.py              # FastAPI application (API + static serving)
├── artifacts/          # Trained model pipeline & metrics
├── data/               # Dataset
├── notebooks/          # EDA & experimentation
├── src/                # Training / preprocessing code
├── static/             # Dashboard (index.html, style.css, main.js)
├── Dockerfile
├── requirements.txt
└── .gitignore
```

## 🚀 Getting Started

### Run locally

```bash
git clone https://github.com/domaamohamed00-ui/StaySignal-Enterprise-AI.git
cd StaySignal-Enterprise-AI

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app:app --reload --port 8000
```

Open <http://localhost:8000> for the dashboard and <http://localhost:8000/docs> for Swagger.

### Run with Docker

```bash
docker build -t staysignal .
docker run -p 8000:8000 staysignal
```

### Deploy on Railway

Push to GitHub, create a new Railway project from the repo – it detects the `Dockerfile` and injects `PORT` automatically.

## 🛠️ Tech Stack

**ML:** XGBoost · scikit-learn · pandas · NumPy · joblib  
**Backend:** FastAPI · Uvicorn · Pydantic  
**Frontend:** HTML · CSS · Vanilla JavaScript  
**DevOps:** Docker · GitHub · Railway

## 🗺️ Roadmap

- [ ] Pin dependency versions in `requirements.txt`
- [ ] Unit tests + GitHub Actions CI
- [ ] Docker `HEALTHCHECK` and non-root user
- [ ] Prometheus/Grafana monitoring & drift detection
- [ ] SHAP-based per-booking explanations
- [ ] Batch scoring endpoint (CSV upload)

## 👤 Author

GitHub: [@domaamohamed00-ui](https://github.com/domaamohamed00-ui)

---

<div align="center">⭐ If you find this project useful, consider giving it a star.</div>
