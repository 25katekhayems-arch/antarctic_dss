"""
MAIN PIPELINE - runs every module in order and returns ONE combined result.
"""
import json
import numpy as np

from region import REGION, PORT_A, PORT_B, generate_sic_history
from predict_sea_ice import predict_sea_ice
from predict_iceberg import predict_iceberg_trajectories, DETECTED_ICEBERGS
from collision_risk import compute_risk_grid
from route_optimizer import find_safe_route

VESSEL_START = PORT_A
FORECAST_DAY = 0


def run_pipeline(start_lonlat=None, end_lonlat=None):
    if start_lonlat is None:
        start_lonlat = (PORT_A["lon"], PORT_A["lat"])
    if end_lonlat is None:
        end_lonlat = (PORT_B["lon"], PORT_B["lat"])

    # 1. Satellite-based sea-ice visualization (raw recent history)
    sic_history = generate_sic_history(n_days=15)

    # 2 & 4. Sea-ice concentration prediction (ML ensemble + numpy forecast)
    sic_result = predict_sea_ice(sic_history)
    sic_forecast = sic_result["forecast"]
    ml_uncertainty = sic_result.get("uncertainty")
    ml_used = sic_result.get("ml_used", False)

    # 2 (iceberg detection) & 3. Iceberg trajectory prediction
    iceberg_trajectories = predict_iceberg_trajectories(icebergs=DETECTED_ICEBERGS)

    # 6. Collision-risk calculation (incorporate ML uncertainty if available)
    risk_grid = compute_risk_grid(sic_forecast[FORECAST_DAY], iceberg_trajectories, day_index=FORECAST_DAY)

    # 7. Safe-route recommendation
    route = find_safe_route(
        risk_grid,
        start_lonlat=start_lonlat,
        end_lonlat=end_lonlat,
    )

    fuel_estimate = None
    if route["success"]:
        FUEL_PER_DEG_OPEN_WATER = 180.0
        SPEED_DEG_PER_HR = 0.35
        fuel_estimate = {
            "estimated_fuel_liters": round(route["routing_cost"] * FUEL_PER_DEG_OPEN_WATER / 9.0, 1),
            "estimated_time_hours": round(route["n_waypoints"] * (SPEED_DEG_PER_HR ** -1) * 0.1, 1),
        }

    # Load ML metrics if available
    ml_metrics = None
    if ml_used:
        try:
            from ml_model import METRICS_PATH
            with open(METRICS_PATH) as f:
                ml_metrics = json.load(f)
        except Exception:
            pass

    result = {
        "region": REGION,
        "vessel_position": {"lon": start_lonlat[0], "lat": start_lonlat[1]},
        "destination": {"lon": end_lonlat[0], "lat": end_lonlat[1]},
        "fuel_estimate": fuel_estimate,
        "sic_forecast_summary": {
            f"day+{i+1}": {
                "mean_sic_pct": round(float(sic_forecast[i].mean()), 1),
                "max_sic_pct": round(float(sic_forecast[i].max()), 1),
            }
            for i in range(sic_forecast.shape[0])
        },
        "iceberg_predictions": {
            name: {"start": [t["lons"][0], t["lats"][0]], "end": [t["lons"][-1], t["lats"][-1]]}
            for name, t in iceberg_trajectories.items()
        },
        "iceberg_summary": {
            "total_count": len(DETECTED_ICEBERGS),
            "total_area_km2": sum(b["length"] * b["width"] / 1e6 for b in DETECTED_ICEBERGS),
            "icebergs": [
                {
                    "name": b["name"],
                    "length_m": b["length"],
                    "width_m": b["width"],
                    "sail_height_m": b["sail"],
                    "draft_depth_m": b["draft"],
                    "area_km2": round(b["length"] * b["width"] / 1e6, 4),
                    "category": (
                        "GIANT" if b["length"] > 10000 else
                        "LARGE" if b["length"] > 2000 else
                        "MEDIUM" if b["length"] > 500 else
                        "SMALL" if b["length"] > 200 else
                        "BERGY BIT"
                    ),
                }
                for b in DETECTED_ICEBERGS
            ],
        },
        "risk_summary": {
            "pct_region_high_risk": round(float((risk_grid > 0.5).mean() * 100), 1),
        },
        "ml_model": {
            "active": ml_used,
            "uncertainty_mean_pct": round(float(ml_uncertainty.mean()), 2) if ml_uncertainty is not None else None,
            "uncertainty_std_pct": round(float(ml_uncertainty.std()), 2) if ml_uncertainty is not None else None,
            "metrics": ml_metrics,
        },
        "recommended_route": route,
    }

    np.save("output_sic_history.npy", sic_history)
    np.save("output_sic_forecast.npy", sic_forecast)
    np.save("output_risk_grid.npy", risk_grid)
    with open("output_iceberg_trajectories.json", "w") as f:
        json.dump(iceberg_trajectories, f)
    with open("output_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
