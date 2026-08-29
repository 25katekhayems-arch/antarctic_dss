# 🧊 N-GEO — Antarctic Decision Support System

A real-time navigation and route-planning system for Antarctic waters — combining satellite sea-ice data, ML-powered forecasting, iceberg tracking, and collision-risk analysis into a single interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **Sea Ice Forecasting** — 5-day SIC (Sea Ice Concentration) prediction using Random Forest ensemble + numpy extrapolation
- **Iceberg Tracking** — Trajectory prediction with drift direction, speed, and threat classification (CRITICAL / HIGH / MODERATE)
- **Collision Risk Grid** — 80×80 cell risk map combining ice density and iceberg proximity
- **AI-Optimized Routing** — A*-based pathfinding through the risk grid with fuel and transit-time estimates
- **Interactive 2D Map** — Leaflet.js dashboard with heatmap overlays, clickable route planning, and live data panels
- **3D Visualization** — Three.js ocean view with animated ship, icebergs, and wave effects
- **Real Satellite Data** — Pulls live SIC from Copernicus Marine Service (OSI SAF AMSR2), with synthetic fallback

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (HTML/JS)                   │
│   Leaflet Map  │  Chart.js  │  Three.js 3D Viewer       │
└────────────────────────┬────────────────────────────────┘
                         │ REST API
┌────────────────────────┴────────────────────────────────┐
│                  FastAPI Backend (api.py)                 │
├──────────┬───────────┬───────────────┬──────────────────┤
│ Sea Ice  │ Iceberg   │ Collision     │ Route            │
│ Forecast │ Tracker   │ Risk Engine   │ Optimizer        │
├──────────┴───────────┴───────────────┴──────────────────┤
│              ML Ensemble (scikit-learn)                   │
├─────────────────────────────────────────────────────────┤
│        Copernicus Marine Service (real satellite data)   │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/25katekhayems-arch/antarctic_dss.git
cd antarctic_dss
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open in browser

```
http://localhost:8000
```

The dashboard loads with the map, forecasts, and iceberg data immediately.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive map dashboard |
| `/health` | GET | Health check |
| `/recommend_route` | GET | Full pipeline result with route recommendation |
| `/recommend_route?start_lon=&start_lat=&end_lon=&end_lat=` | GET | Custom route between two points |
| `/sic_grid?day=0` | GET | Sea ice concentration grid for a forecast day (0–4) |
| `/risk_grid` | GET | Collision risk grid |
| `/iceberg_trajectories` | GET | Iceberg drift paths |
| `/iceberg_details` | GET | Detailed iceberg info (speed, direction, threat) |
| `/historical_analysis` | GET | Seasonal SIC trends and statistics |
| `/weather_buoys` | GET | Synthetic weather buoy data |
| `/ml_metrics` | GET | ML model training metrics |
| `/train_model` | POST | Retrain the ML ensemble |

---

## 🗂️ Project Structure

```
antarctic_dss/
├── frontend/
│   ├── index.html          # Main 2D map dashboard
│   └── 3d_viewer.html      # Three.js 3D ocean view
├── api.py                  # FastAPI server (serves API + frontend)
├── main.py                 # Pipeline orchestrator
├── region.py               # Region definition + real Copernicus data fetcher
├── predict_sea_ice.py      # SIC forecasting (ML + numpy)
├── predict_iceberg.py      # Iceberg trajectory prediction
├── collision_risk.py       # Risk grid computation
├── route_optimizer.py      # A* pathfinding through risk grid
├── ml_model.py             # Random Forest ensemble model
├── ml_trainer.py           # Training pipeline for the ML model
├── model.py                # Data model definitions
├── train.py                # CLI training script
├── requirements.txt        # Python dependencies
├── render.yaml             # Render.com deployment config
└── .gitignore
```

---

## 🌊 Real Satellite Data (Optional)

The system can pull live Sea Ice Concentration data from the **Copernicus Marine Service**:

1. Register free at [marine.copernicus.eu](https://marine.copernicus.eu)
2. Install the toolbox: `pip install copernicusmarine`
3. Login: `python -c "import copernicusmarine; copernicusmarine.login()"`
4. Restart the server — real data loads automatically

Without credentials, the system uses high-quality synthetic data that closely mimics real Antarctic conditions.

---

## 🧪 ML Model

The sea ice forecast uses a **Random Forest ensemble** trained on historical SIC patterns:

- **Algorithm**: Random Forest Regressor (scikit-learn)
- **Features**: Spatial SIC gradients, temporal trends, seasonal patterns
- **Output**: 5-day SIC forecast with uncertainty estimates
- **Retrain**: `POST /train_model` or run `python train.py`

---

## 🌐 Live Deployment

This project is deployed on **Render.com** (free tier):

🔗 [Live Demo](https://n-geo.onrender.com/)

> Free tier spins down after 15 min of inactivity. First request may take ~30s to wake up.

---

## 📋 Requirements

- Python 3.11+
- No GPU required (CPU inference is fast enough for real-time)

---

## 📄 License

MIT

---

*Built for Antarctic maritime safety research.*
