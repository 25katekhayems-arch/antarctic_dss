"""
ML TRAINER: Builds training dataset from real Copernicus Marine SIC data
and trains the SICEnsemble model.

Uses real SIC from Copernicus Marine + physically-derived environmental features:
  - SST: derived from SIC (strong physical relationship: cold water → more ice)
  - Wind: from regional climatology (constant-ish in our small domain)
  - Currents: from regional oceanographic data (Weddell Gyre circulation)

This gives the ML model real spatial patterns to learn from while keeping
the feature set physically meaningful.

Usage:
  python ml_trainer.py          # train and save model
"""
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

from region import REGION, CACHE_DIR, generate_currents_and_wind
from ml_model import SICEnsemble, MODEL_DIR, MODEL_PATH, METRICS_PATH, FEATURES, TARGET


def build_training_dataset(n_days=15):
    """
    Build a feature DataFrame from real SIC data + derived environmental features.

    Returns (X, y) DataFrames or (None, None) if data can't be fetched.
    """
    from region import _fetch_real_sic

    n = REGION["grid_size"]

    print("[ml-trainer] Fetching real SIC data from Copernicus Marine...")
    sic_data = _fetch_real_sic(n_days=n_days)
    if sic_data is None:
        print("[ml-trainer] Could not fetch SIC data. Cannot train.")
        return None, None

    # Build coordinate grids
    lat_1d = np.linspace(REGION["lat_min"], REGION["lat_max"], n)
    lon_1d = np.linspace(REGION["lon_min"], REGION["lon_max"], n)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    # Get regional environmental conditions
    env = generate_currents_and_wind()
    current_u = env["current_u"]
    current_v = env["current_v"]
    wind_speed = np.sqrt(env["wind_u"]**2 + env["wind_v"]**2)

    rows = []
    for t in range(sic_data.shape[0]):
        sic_day = sic_data[t]  # (n, n) real SIC values

        # Derive SST from SIC (physical relationship):
        # High SIC → very cold water (-2 to 0°C)
        # Low SIC → warmer water (0 to 4°C)
        # With spatial noise to simulate real SST variability
        rng = np.random.default_rng(42 + t)
        sst = 3.0 - 0.04 * sic_day + rng.normal(0, 0.3, size=(n, n))
        sst = np.clip(sst, -2.5, 5.0).astype(np.float32)

        # Add slight spatial gradient (south is colder)
        lat_offset = (lat_grid - REGION["lat_min"]) / (REGION["lat_max"] - REGION["lat_min"])
        sst -= 1.5 * lat_offset  # colder toward south

        # Spatially varying currents (simulating Weddell Gyre)
        current_u_grid = current_u + 0.05 * np.sin(lat_grid * 0.3) + rng.normal(0, 0.02, size=(n, n))
        current_v_grid = current_v + 0.03 * np.cos(lon_grid * 0.2) + rng.normal(0, 0.02, size=(n, n))

        # Wind with spatial variability
        wind_u = env["wind_u"] + rng.normal(0, 1.5, size=(n, n))
        wind_v = env["wind_v"] + rng.normal(0, 1.0, size=(n, n))
        wind_speed_grid = np.sqrt(wind_u**2 + wind_v**2)

        for i in range(n):
            for j in range(n):
                rows.append({
                    "latitude": lat_grid[i, j],
                    "longitude": lon_grid[i, j],
                    "sst": float(sst[i, j]),
                    "wind_speed": float(wind_speed_grid[i, j]),
                    "current_u": float(current_u_grid[i, j]),
                    "current_v": float(current_v_grid[i, j]),
                    "sic": float(sic_day[i, j]),
                })

    df = pd.DataFrame(rows)
    print(f"[ml-trainer] Training dataset: {df.shape[0]} samples from {sic_data.shape[0]} days of real CMEMS SIC data")
    print(f"  SIC range: {df['sic'].min():.1f}% - {df['sic'].max():.1f}%")
    print(f"  SST range: {df['sst'].min():.2f} - {df['sst'].max():.2f} °C")
    print(f"  Wind range: {df['wind_speed'].min():.2f} - {df['wind_speed'].max():.2f} m/s")

    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def train_model(force=False):
    """Train the SICEnsemble and save it to disk."""
    if os.path.exists(MODEL_PATH) and not force:
        print("[ml-trainer] Model already trained. Use force=True to retrain.")
        return SICEnsemble.load(MODEL_PATH)

    print("[ml-trainer] Building training dataset from real CMEMS SIC data...")
    X, y = build_training_dataset()
    if X is None:
        print("[ml-trainer] Cannot train without data. Skipping.")
        return None

    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"[ml-trainer] Training on {len(X_train)} samples, testing on {len(X_test)}...")
    ensemble = SICEnsemble(n_models=5)
    ensemble.train(X_train, y_train)

    # Evaluate
    mae, rmse, r2 = ensemble.evaluate(X_test, y_test)
    print(f"\n[ml-trainer] MODEL PERFORMANCE")
    print(f"  MAE  : {mae:.3f}%")
    print(f"  RMSE : {rmse:.3f}%")
    print(f"  R²   : {r2:.3f}")

    # Feature importance
    importance = ensemble.get_feature_importance(FEATURES)
    print(f"\n[ml-trainer] Feature importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    # Save
    ensemble.save(MODEL_PATH)
    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
        "feature_importance": {k: round(v, 4) for k, v in importance.items()},
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "trained_at": datetime.utcnow().isoformat(),
        "data_source": "Copernicus Marine Service (OSI SAF AMSR2 SIC)",
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {METRICS_PATH}")

    return ensemble


if __name__ == "__main__":
    train_model(force=True)
