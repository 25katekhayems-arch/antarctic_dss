"""
MODULE: Sea-ice concentration prediction.

Input:  recent SIC history for the region
Output: predicted SIC maps for the next N days, plus uncertainty estimates

Uses a two-stage approach:
  1. ML Ensemble (Random Forest): predicts current SIC from environmental
     features (lat, lon, SST, wind, currents) with uncertainty quantification
  2. Numpy persistence model: extrapolates the ML prediction forward in time
     using diffusion and trend analysis
"""
import numpy as np
from model import forecast_sic
from region import REGION, grid_to_lonlat

SEQ_LEN = 10       # how many recent days to feed the forecaster
FORECAST_LEN = 5

# Global state for ML model
_ml_model = None
_ml_loaded = False


def _load_ml_model():
    """Load the trained ML ensemble model (cached)."""
    global _ml_model, _ml_loaded
    if _ml_loaded:
        return _ml_model
    _ml_loaded = True
    try:
        from ml_model import SICEnsemble
        _ml_model = SICEnsemble.load()
        if _ml_model is not None:
            print("[predict_sea_ice] Loaded ML ensemble model")
    except Exception as e:
        print(f"[predict_sea_ice] Could not load ML model: {e}")
    return _ml_model


def predict_sic_current(ml_model, sic_history):
    """
    Use the ML ensemble to predict current SIC for every grid cell,
    along with uncertainty bounds.

    Returns:
      sic_current: (H, W) predicted SIC for the latest time step
      uncertainty: (H, W) uncertainty (std) across ensemble models
      sic_lower:   (H, W) lower bound of 95% CI
      sic_upper:   (H, W) upper bound of 95% CI
    """
    import pandas as pd
    from region import REGION, generate_currents_and_wind

    n = REGION["grid_size"]
    lat_1d = np.linspace(REGION["lat_min"], REGION["lat_max"], n)
    lon_1d = np.linspace(REGION["lon_min"], REGION["lon_max"], n)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    env = generate_currents_and_wind()
    current_u = env["current_u"]
    current_v = env["current_v"]
    wind_u = env["wind_u"]
    wind_v = env["wind_v"]
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)

    # Approximate SST from SIC (cold water = high ice)
    last_sic = sic_history[-1]
    sst_approx = 2.0 - 0.03 * last_sic  # rough SST-SIC relationship

    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({
                "latitude": lat_grid[i, j],
                "longitude": lon_grid[i, j],
                "sst": float(sst_approx[i, j]),
                "wind_speed": float(wind_speed),
                "current_u": float(current_u),
                "current_v": float(current_v),
            })

    X = pd.DataFrame(rows)
    result = ml_model.predict(X)

    sic_current = result["prediction"].reshape(n, n)
    uncertainty = result["uncertainty"].reshape(n, n)
    sic_lower = result["lower"].reshape(n, n)
    sic_upper = result["upper"].reshape(n, n)

    return sic_current, uncertainty, sic_lower, sic_upper


def predict_sea_ice(sic_history, forecast_len=FORECAST_LEN, model_path="sic_forecaster.pt"):
    """
    sic_history: (n_days>=SEQ_LEN, H, W) array, most recent day last
    returns: dict with:
      - forecast: (forecast_len, H, W) array of predicted SIC (%)
      - uncertainty: (H, W) ensemble uncertainty for current SIC
      - ml_used: whether ML model was used
    """
    recent = sic_history[-SEQ_LEN:]  # (SEQ_LEN, H, W)

    # Try ML model for current SIC refinement
    ml_model = _load_ml_model()
    ml_used = False
    uncertainty = None

    if ml_model is not None:
        try:
            ml_sic, uncertainty, ml_lower, ml_upper = predict_sic_current(ml_model, recent)
            # Blend ML prediction with last observed (80% ML, 20% persistence)
            blended = 0.8 * ml_sic + 0.2 * recent[-1]
            blended = np.clip(blended, 0, 100).astype(np.float32)

            # Replace the last frame with the ML-blended version before forecasting
            enhanced_history = np.concatenate([recent[:-1], blended[None, :, :]], axis=0)
            forecast = forecast_sic(enhanced_history, forecast_len=forecast_len)
            ml_used = True
            print(f"[predict_sea_ice] ML model active — uncertainty range: "
                  f"{uncertainty.mean():.2f}% ±{uncertainty.std():.2f}%")
        except Exception as e:
            print(f"[predict_sea_ice] ML prediction failed, using numpy model: {e}")
            forecast = forecast_sic(recent, forecast_len=forecast_len)
    else:
        forecast = forecast_sic(recent, forecast_len=forecast_len)

    return {
        "forecast": forecast,
        "uncertainty": uncertainty,
        "ml_used": ml_used,
    }


if __name__ == "__main__":
    from region import generate_sic_history
    history = generate_sic_history(n_days=15)
    result = predict_sea_ice(history)
    forecast = result["forecast"]
    print("Forecast shape:", forecast.shape)
    print("Day+1 mean SIC:", round(float(forecast[0].mean()), 1), "%")
    print("Day+5 mean SIC:", round(float(forecast[-1].mean()), 1), "%")
    print("ML model used:", result["ml_used"])
    if result["uncertainty"] is not None:
        print("Mean uncertainty:", round(float(result["uncertainty"].mean()), 2), "%")
    np.save("sic_forecast_output.npy", forecast)
    np.save("sic_history_output.npy", history)
