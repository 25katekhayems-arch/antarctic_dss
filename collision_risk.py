"""
MODULE: Collision-risk calculation.

Input:  predicted SIC grid + predicted iceberg trajectories + vessel max-ice-tolerance
Output: a single "risk grid" (H, W), 0 = safe, 1 = impassable/high-risk

This is the fusion step: it turns two separate forecasts (ice concentration,
iceberg position) into ONE map the route optimizer can consume.
"""
import numpy as np
from region import REGION, lonlat_to_grid

MAX_SAFE_SIC = 70          # vessel treats >70% concentration as high-risk
ICEBERG_BUFFER_CELLS = 2   # mark cells within this radius of a predicted iceberg position as risky


def compute_risk_grid(sic_forecast_day, iceberg_trajectories, day_index=0):
    """
    sic_forecast_day: (H, W) SIC forecast for one specific day
    iceberg_trajectories: dict from predict_iceberg_trajectories()
    day_index: which step of the iceberg trajectory corresponds to "this day"
    returns: (H, W) risk grid, values 0.0 (safe) to 1.0 (max risk)
    """
    n = REGION["grid_size"]
    risk = np.zeros((n, n), dtype=np.float32)

    # --- ice concentration contribution ---
    ice_risk = np.clip((sic_forecast_day - MAX_SAFE_SIC) / (100 - MAX_SAFE_SIC), 0, 1)
    risk = np.maximum(risk, ice_risk)

    # --- iceberg proximity contribution ---
    steps_per_day = 96  # 900s steps -> 96 steps/day
    for name, traj in iceberg_trajectories.items():
        step = min(day_index * steps_per_day, len(traj["lons"]) - 1)
        lon, lat = traj["lons"][step], traj["lats"][step]
        if np.isnan(lon) or np.isnan(lat):
            continue
        row, col = lonlat_to_grid(lon, lat)
        r0, r1 = max(0, row - ICEBERG_BUFFER_CELLS), min(n, row + ICEBERG_BUFFER_CELLS + 1)
        c0, c1 = max(0, col - ICEBERG_BUFFER_CELLS), min(n, col + ICEBERG_BUFFER_CELLS + 1)
        risk[r0:r1, c0:c1] = 1.0

    return risk


if __name__ == "__main__":
    from predict_sea_ice import predict_sea_ice
    from predict_iceberg import predict_iceberg_trajectories
    from region import generate_sic_history

    history = generate_sic_history()
    sic_forecast = predict_sea_ice(history)
    trajectories = predict_iceberg_trajectories()

    risk_day1 = compute_risk_grid(sic_forecast[0], trajectories, day_index=0)
    print("Risk grid shape:", risk_day1.shape)
    print("Fraction of region high-risk (>0.5):", round(float((risk_day1 > 0.5).mean()), 3))
    np.save("risk_grid_day1.npy", risk_day1)
