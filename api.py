"""
Backend API for the Antarctic DSS.

Serves BOTH the REST API and the frontend website from a single port.

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000 - the map dashboard loads directly.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from main import run_pipeline
from region import REGION

app = FastAPI(title="Antarctic DSS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {"result": None}


@app.on_event("startup")
def warmup():
    print("Warming up pipeline...")
    _cache["result"] = run_pipeline()
    print("Warmup complete.")


@app.get("/health")
def health():
    return {"status": "ok", "region": REGION["name"]}


@app.get("/recommend_route")
def recommend_route(
    refresh: bool = False,
    start_lon: float = None,
    start_lat: float = None,
    end_lon: float = None,
    end_lat: float = None,
):
    has_custom_points = None not in (start_lon, start_lat, end_lon, end_lat)

    if has_custom_points:
        return run_pipeline(
            start_lonlat=(start_lon, start_lat),
            end_lonlat=(end_lon, end_lat),
        )

    if _cache["result"] is None or refresh:
        _cache["result"] = run_pipeline()
    return _cache["result"]


@app.get("/risk_grid")
def risk_grid():
    import numpy as np
    grid = np.load("output_risk_grid.npy")
    return {
        "grid": grid.tolist(),
        "bounds": {
            "lon_min": REGION["lon_min"], "lon_max": REGION["lon_max"],
            "lat_min": REGION["lat_min"], "lat_max": REGION["lat_max"],
        },
    }


@app.get("/sic_grid")
def sic_grid(day: int = 0):
    import numpy as np
    forecast = np.load("output_sic_forecast.npy")
    day = max(0, min(day, forecast.shape[0] - 1))
    return {
        "day": day + 1,
        "grid": forecast[day].tolist(),
        "bounds": {
            "lon_min": REGION["lon_min"], "lon_max": REGION["lon_max"],
            "lat_min": REGION["lat_min"], "lat_max": REGION["lat_max"],
        },
    }


@app.get("/iceberg_trajectories")
def iceberg_trajectories():
    import json
    with open("output_iceberg_trajectories.json") as f:
        return json.load(f)


@app.get("/iceberg_details")
def iceberg_details():
    """Returns detailed iceberg information for the dashboard."""
    import json
    import numpy as np
    from region import REGION
    
    with open("output_iceberg_trajectories.json") as f:
        trajectories = json.load(f)
    
    with open("output_risk_grid.npy", "rb") as f:
        risk_grid = np.load(f)
    
    iceberg_details = []
    for name, traj in trajectories.items():
        lons = traj["lons"]
        lats = traj["lats"]
        
        # Calculate displacement
        start_lon, start_lat = lons[0], lats[0]
        end_lon, end_lat = lons[-1], lats[-1]
        
        # Distance in km (Haversine approximation)
        R = 6371.0
        lat1, lat2 = np.radians(start_lat), np.radians(end_lat)
        dlat = lat2 - lat1
        dlon = np.radians(end_lon - start_lon)
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        distance_km = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        # Drift direction (bearing)
        dlon_rad = np.radians(end_lon - start_lon)
        y = np.sin(dlon_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon_rad)
        bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
        
        # Average speed (km/h over 96 hours)
        speed_kmh = distance_km / 96.0 if distance_km > 0 else 0
        
        # Distance from route center
        center_lon = (REGION["lon_min"] + REGION["lon_max"]) / 2
        center_lat = (REGION["lat_min"] + REGION["lat_max"]) / 2
        dlon_c = np.radians(start_lon - center_lon)
        lat1_c = np.radians(center_lat)
        lat2_c = np.radians(start_lat)
        a_c = np.sin((lat2_c-lat1_c)/2)**2 + np.cos(lat1_c) * np.cos(lat2_c) * np.sin(dlon_c/2)**2
        dist_from_center = R * 2 * np.arctan2(np.sqrt(a_c), np.sqrt(1-a_c))
        
        # Risk level at current position
        row = int((start_lat - REGION["lat_min"]) / (REGION["lat_max"] - REGION["lat_min"]) * (risk_grid.shape[0]-1))
        col = int((start_lon - REGION["lon_min"]) / (REGION["lon_max"] - REGION["lon_min"]) * (risk_grid.shape[1]-1))
        row = max(0, min(row, risk_grid.shape[0]-1))
        col = max(0, min(col, risk_grid.shape[1]-1))
        risk_value = float(risk_grid[row, col])
        
        iceberg_details.append({
            "name": name,
            "current_lon": round(start_lon, 3),
            "current_lat": round(start_lat, 3),
            "predicted_end_lon": round(end_lon, 3),
            "predicted_end_lat": round(end_lat, 3),
            "displacement_km": round(distance_km, 2),
            "drift_direction_deg": round(bearing, 1),
            "drift_direction_compass": _degrees_to_compass(bearing),
            "avg_speed_kmh": round(speed_kmh, 3),
            "avg_speed_knots": round(speed_kmh * 0.5399568, 3),
            "dist_from_center_km": round(dist_from_center, 2),
            "risk_level": round(risk_value, 3),
            "risk_category": _categorize_risk(risk_value),
            "trajectory_length": len(lons),
            "forecast_hours": 96
        })
    
    return {
        "total_icebergs": len(iceberg_details),
        "icebergs": iceberg_details,
        "region": REGION["name"],
        "forecast_window_hours": 96
    }


def _degrees_to_compass(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return dirs[idx]


def _categorize_risk(risk_value):
    if risk_value >= 0.7:
        return "CRITICAL"
    elif risk_value >= 0.4:
        return "HIGH"
    elif risk_value >= 0.2:
        return "MODERATE"
    else:
        return "LOW"


@app.get("/historical_analysis")
def historical_analysis():
    """Returns historical SIC trends and regional statistics for the dashboard."""
    import json
    import numpy as np
    from region import REGION, generate_sic_history
    
    history = generate_sic_history(n_days=120)
    
    # Monthly averages (each row is 30 days)
    monthly_avgs = []
    monthly_mins = []
    monthly_maxs = []
    for i in range(0, history.shape[0], 30):
        chunk = history[i:i+30]
        monthly_avgs.append(round(float(np.mean(chunk)), 2))
        monthly_mins.append(round(float(np.min(chunk)), 2))
        monthly_maxs.append(round(float(np.max(chunk)), 2))
    
    # Seasonal pattern
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    seasonal_pattern = {
        "months": month_names[:len(monthly_avgs)],
        "avg_sic": monthly_avgs,
        "min_sic": monthly_mins,
        "max_sic": monthly_maxs,
    }
    
    # Current SIC stats
    current = history[-1]
    stats = {
        "mean": round(float(np.mean(current)), 2),
        "std": round(float(np.std(current)), 2),
        "min": round(float(np.min(current)), 2),
        "max": round(float(np.max(current)), 2),
        "pct_above_50": round(float(np.mean(current > 50) * 100), 1),
        "pct_above_80": round(float(np.mean(current > 80) * 100), 1),
        "pct_above_95": round(float(np.mean(current > 95) * 100), 1),
    }
    
    # Trend (linear regression over monthly averages)
    trend = {"slope_per_month": 0, "direction": "stable", "annual_change_pct": 0}
    if len(monthly_avgs) >= 2:
        x = np.arange(len(monthly_avgs), dtype=float)
        y = np.array(monthly_avgs, dtype=float)
        try:
            coeffs = np.polyfit(x, y, 1)
            trend = {
                "slope_per_month": round(float(coeffs[0]), 4),
                "direction": "increasing" if coeffs[0] > 0.01 else "decreasing" if coeffs[0] < -0.01 else "stable",
                "annual_change_pct": round(float(coeffs[0] * 12), 2),
            }
        except np.linalg.LinAlgError:
            pass
    
    return {
        "region": REGION["name"],
        "history_days": int(history.shape[0]),
        "current_sic_stats": stats,
        "seasonal_pattern": seasonal_pattern,
        "trend": trend,
        "ice_concentration_regimes": {
            "open_water": round(float(np.mean(current < 15)), 3),
            "marginal": round(float(np.mean((current >= 15) & (current < 50))), 3),
            "moderate": round(float(np.mean((current >= 50) & (current < 80))), 3),
            "dense": round(float(np.mean((current >= 80) & (current < 95))), 3),
            "compact": round(float(np.mean(current >= 95)), 3),
        },
    }


@app.get("/ml_metrics")
def ml_metrics():
    """Returns ML model training metrics if available."""
    try:
        from ml_model import METRICS_PATH
        import json
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {"active": False, "message": "ML model not trained yet. POST /train_model to train."}


@app.post("/train_model")
def train_model_endpoint():
    """Train the ML ensemble on real Copernicus Marine data."""
    try:
        from ml_trainer import train_model
        ensemble = train_model(force=True)
        if ensemble is not None:
            # Re-run pipeline with new model
            _cache["result"] = run_pipeline()
            return {"status": "success", "message": "ML model trained and pipeline re-run."}
        else:
            return {"status": "error", "message": "Training failed. Check credentials."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/weather_buoys")
def weather_buoys():
    """Returns synthetic weather buoy positions with environmental data."""
    import numpy as np
    from region import REGION, generate_sic_history
    from datetime import datetime

    rng = np.random.default_rng(42)
    sic_history = generate_sic_history(n_days=15)
    last_sic = sic_history[-1]
    n = REGION["grid_size"]

    # Place 12 buoys across the region in a grid-like pattern with jitter
    buoys = []
    buoy_id = 1
    for bi in range(3):
        for bj in range(4):
            row = int(5 + bi * (n - 10) / 2 + rng.uniform(-3, 3))
            col = int(5 + bj * (n - 10) / 3 + rng.uniform(-3, 3))
            row = max(0, min(row, n - 1))
            col = max(0, min(col, n - 1))
            from region import grid_to_lonlat
            lon, lat = grid_to_lonlat(row, col)

            sic_val = float(last_sic[row, col])
            sst = round(2.0 - 0.025 * sic_val + rng.normal(0, 0.3), 1)
            wind_speed = round(8.0 + rng.uniform(0, 8), 1)
            wind_dir = round(rng.uniform(0, 360), 0)
            current_speed = round(0.05 + rng.uniform(0, 0.15), 2)
            current_dir = round(180 + rng.normal(0, 30), 0) % 360
            wave_height = round(1.5 + wind_speed * 0.15 + rng.normal(0, 0.3), 1)
            pressure = round(995 + rng.normal(0, 5), 1)
            humidity = round(75 + rng.normal(0, 8), 0)
            visibility = round(max(0.5, 10 - sic_val * 0.08 + rng.normal(0, 1)), 1)

            def _compass(d):
                dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
                        "S","SSW","SW","WSW","W","WNW","NW","NNW"]
                return dirs[round(d / 22.5) % 16]

            buoys.append({
                "id": f"WB-{buoy_id:02d}",
                "lon": round(lon, 3),
                "lat": round(lat, 3),
                "sst_c": sst,
                "wind_speed_ms": wind_speed,
                "wind_dir_deg": int(wind_dir),
                "wind_dir_compass": _compass(wind_dir),
                "current_speed_ms": current_speed,
                "current_dir_deg": int(current_dir),
                "current_dir_compass": _compass(current_dir),
                "wave_height_m": wave_height,
                "pressure_hpa": pressure,
                "humidity_pct": int(humidity),
                "visibility_km": visibility,
                "sic_pct": round(sic_val, 1),
                "status": "ACTIVE" if rng.random() > 0.1 else "DEGRADED",
                "last_update": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            })
            buoy_id += 1

    return {"total": len(buoys), "buoys": buoys}


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
