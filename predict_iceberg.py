"""
MODULE: Iceberg trajectory prediction.

Input:  detected iceberg starting positions + sizes, region's current/wind
Output: predicted (lon, lat) trajectory for each iceberg over the forecast window

Uses a simple kinematic drift model: icebergs move with ocean current + a
fraction of wind (windage factor ~0.03-0.05 typical for icebergs).  No heavy
external dependencies required.
"""
import numpy as np
from datetime import datetime, timedelta
from region import generate_currents_and_wind


# Example "detected" icebergs for the demo region (in real system: from SAR detection)
DETECTED_ICEBERGS = [
    {"name": "Berg-1", "lon": -60.0, "lat": -62.0, "length": 1200, "width": 500, "sail": 30, "draft": 150},
    {"name": "Berg-2", "lon": -55.0, "lat": -63.5, "length": 250,  "width": 120, "sail": 15, "draft": 90},
    {"name": "Berg-3", "lon": -45.0, "lat": -66.0, "length": 60,   "width": 30,  "sail": 8,  "draft": 35},
]

# Windage factor: fraction of wind speed that drives iceberg drift
# Smaller icebergs have higher windage because more of them is above water
WINDAGE_BASE = 0.03


def predict_iceberg_trajectories(icebergs=DETECTED_ICEBERGS, forecast_hours=96, start_time=None):
    """
    Returns a dict: {iceberg_name: {"lons": [...], "lats": [...], "times": [...]}}
    """
    env = generate_currents_and_wind()
    cur_u = env["current_u"]  # m/s
    cur_v = env["current_v"]  # m/s
    wind_u = env["wind_u"]    # m/s
    wind_v = env["wind_v"]    # m/s

    dt_s = 900  # 15 min time step
    n_steps = int(forecast_hours * 3600 / dt_s)

    results = {}
    for berg in icebergs:
        lon0 = berg["lon"]
        lat0 = berg["lat"]
        size_factor = max(0.5, min(2.0, 500.0 / max(berg["length"], 1)))
        windage = WINDAGE_BASE * size_factor

        lons = [lon0]
        lats = [lat0]
        lon, lat = lon0, lat0

        for s in range(n_steps):
            vx = cur_u + windage * wind_u
            vy = cur_v + windage * wind_v
            dlat = (vy * dt_s) / 111_320.0
            dlon = (vx * dt_s) / (111_320.0 * max(np.cos(np.radians(lat)), 0.01))
            lon += dlon
            lat += dlat
            lons.append(lon)
            lats.append(lat)

        results[berg["name"]] = {"lons": lons, "lats": lats}
    return results


if __name__ == "__main__":
    import json
    trajectories = predict_iceberg_trajectories()
    for name, t in trajectories.items():
        print(f"{name}: start=({t['lons'][0]:.2f},{t['lats'][0]:.2f}) "
              f"end=({t['lons'][-1]:.2f},{t['lats'][-1]:.2f}) over {len(t['lons'])} steps")
    with open("iceberg_trajectories.json", "w") as f:
        json.dump(trajectories, f)
