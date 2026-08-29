"""
Defines the Antarctic demo region and generates sea-ice concentration data.

This module tries to load REAL satellite SIC data from the Copernicus Marine
Service (OSI SAF near-real-time product) and falls back to synthetic data
if credentials aren't configured or the network is unavailable.

Real data source:
  Product:  OSI SAF Sea Ice Concentration, AMSR2, Southern Hemisphere
  Dataset:  osisaf_obs-si_glo_phy-sic-south_nrt_amsr2_l4_P1D-m
  Access:   Free account at https://marine.copernicus.eu
            (register, then run: python -c "import copernicusmarine; copernicusmarine.login()")

Everything downstream (forecast model, iceberg drift, routing) just consumes
whatever comes out of generate_sic_history(), so swapping this file is the
only thing that changes.
"""
import numpy as np
import os
import json

# --- Region definition (South Atlantic / Weddell Sea "iceberg alley"-style box) ---
REGION = {
    "name": "Fake Demo Region (Antarctic-style)",
    "lon_min": -65.0, "lon_max": -35.0,
    "lat_min": -70.0, "lat_max": -60.0,
    "grid_size": 80,          # 80x80 cells (upgraded from 40 for more detail)
}

# Two ports for the vessel to travel between
PORT_A = {"name": "Port A (open water entry)", "lon": -63.0, "lat": -61.0}
PORT_B = {"name": "Port B (research station)", "lon": -37.0, "lat": -69.0}

# Copernicus Marine dataset for Antarctic SIC
CMEMS_SIC_DATASET = "osisaf_obs-si_glo_phy-sic-south_nrt_amsr2_l4_P1D-m"

# Local cache for downloaded data
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")


def lonlat_to_grid(lon, lat):
    """Convert a real lon/lat into (row, col) grid indices for REGION."""
    n = REGION["grid_size"]
    col = int((lon - REGION["lon_min"]) / (REGION["lon_max"] - REGION["lon_min"]) * (n - 1))
    row = int((lat - REGION["lat_min"]) / (REGION["lat_max"] - REGION["lat_min"]) * (n - 1))
    return np.clip(row, 0, n - 1), np.clip(col, 0, n - 1)


def grid_to_lonlat(row, col):
    n = REGION["grid_size"]
    lon = REGION["lon_min"] + col / (n - 1) * (REGION["lon_max"] - REGION["lon_min"])
    lat = REGION["lat_min"] + row / (n - 1) * (REGION["lat_max"] - REGION["lat_min"])
    return lon, lat


# ---------------------------------------------------------------------------
# Real data fetcher using Copernicus Marine Toolbox
# ---------------------------------------------------------------------------

def _fetch_real_sic(n_days=15):
    """
    Attempt to download real SIC data from Copernicus Marine Service.

    Returns (n_days, H, W) array of SIC values 0-100 on our REGION grid,
    or None if anything goes wrong (no auth, network, etc.).
    """
    cache_file = os.path.join(CACHE_DIR, "sic_real.npy")
    cache_meta = os.path.join(CACHE_DIR, "sic_real_meta.json")

    # Check cache freshness (re-download after 24 hours)
    if os.path.exists(cache_file) and os.path.exists(cache_meta):
        try:
            with open(cache_meta) as f:
                meta = json.load(f)
            import time
            if time.time() - meta.get("timestamp", 0) < 86400:
                data = np.load(cache_file)
                print(f"[real-data] Loaded cached SIC data: {data.shape}")
                return data
        except Exception:
            pass

    try:
        import copernicusmarine
        from datetime import datetime, timedelta
    except ImportError:
        print("[real-data] copernicusmarine not installed, using synthetic data")
        return None

    # Check for credentials BEFORE calling open_dataset (it prompts interactively otherwise)
    username = os.environ.get('COPERNICUSMARINE_SERVICE_USERNAME')
    password = os.environ.get('COPERNICUSMARINE_SERVICE_PASSWORD')
    if not username or not password:
        import base64
        from pathlib import Path
        # copernicusmarine stores creds as base64-encoded INI in a dotfile
        creds_file = Path.home() / '.copernicusmarine' / '.copernicusmarine-credentials'
        if creds_file.exists():
            try:
                import configparser
                raw = base64.b64decode(creds_file.read_text().strip()).decode()
                cfg = configparser.ConfigParser()
                cfg.read_string(raw)
                if cfg.sections():
                    section = cfg[cfg.sections()[0]]
                    username = username or section.get('username', '')
                    password = password or section.get('password', '')
            except Exception:
                pass

    if not username or not password:
        print("[real-data] No Copernicus Marine credentials found.")
        print("[real-data] To use real satellite data, register free at: https://marine.copernicus.eu")
        print("[real-data] Then run: python -c \"import copernicusmarine; copernicusmarine.login()\"")
        return None

    # Date range: last n_days days
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=n_days + 5)  # extra buffer for missing days

    print(f"[real-data] Fetching real SIC data from Copernicus Marine "
          f"({start_dt.date()} to {end_dt.date()})...")

    try:
        # Try to open the dataset with spatial/temporal subsetting
        # The API handles the coordinate conversion from our lon/lat box
        ds = copernicusmarine.open_dataset(
            dataset_id=CMEMS_SIC_DATASET,
            variables=["ice_conc"],
            minimum_longitude=REGION["lon_min"],
            maximum_longitude=REGION["lon_max"],
            minimum_latitude=REGION["lat_min"],
            maximum_latitude=REGION["lat_max"],
            start_datetime=start_dt.strftime("%Y-%m-%d"),
            end_datetime=end_dt.strftime("%Y-%m-%d"),
            username=username,
            password=password,
        )

        # Extract the SIC variable as numpy array
        sic = ds["ice_conc"]  # standard_name = sea_ice_area_fraction, units %
        print(f"[real-data] Dataset shape: {sic.shape}, dims: {sic.dims}")
        print(f"[real-data] Date range: {sic.time.values[0]} to {sic.time.values[-1]}")

        # Take the last n_days days
        sic_vals = sic.values[-n_days:]

        # Convert to percentage (0-100) if values are 0-1
        if sic_vals.max() <= 1.0 + 1e-6:
            sic_vals = sic_vals * 100.0

        # Replace NaN with 0 (open water)
        sic_vals = np.nan_to_num(sic_vals, nan=0.0)

        # Ensure we have exactly n_days
        if sic_vals.shape[0] < n_days:
            print(f"[real-data] Only got {sic_vals.shape[0]} days, need {n_days}")
            return None

        # Resample to our 40x40 grid if the source grid differs (numpy-only)
        target_n = REGION["grid_size"]
        if sic_vals.shape[1] != target_n or sic_vals.shape[2] != target_n:
            orig_h, orig_w = sic_vals.shape[1], sic_vals.shape[2]
            # Simple bilinear resampling using numpy
            row_idx = np.linspace(0, orig_h - 1, target_n)
            col_idx = np.linspace(0, orig_w - 1, target_n)
            resampled = np.zeros((sic_vals.shape[0], target_n, target_n), dtype=np.float32)
            for day in range(sic_vals.shape[0]):
                for r, ri in enumerate(row_idx):
                    ri0 = int(ri)
                    ri1 = min(ri0 + 1, orig_h - 1)
                    rf = ri - ri0
                    for c, ci in enumerate(col_idx):
                        ci0 = int(ci)
                        ci1 = min(ci0 + 1, orig_w - 1)
                        cf = ci - ci0
                        resampled[day, r, c] = (
                            sic_vals[day, ri0, ci0] * (1 - rf) * (1 - cf) +
                            sic_vals[day, ri1, ci0] * rf * (1 - cf) +
                            sic_vals[day, ri0, ci1] * (1 - rf) * cf +
                            sic_vals[day, ri1, ci1] * rf * cf
                        )
            sic_vals = resampled
            print(f"[real-data] Resampled to {sic_vals.shape}")

        sic_vals = np.clip(sic_vals, 0, 100).astype(np.float32)

        # Cache the data
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cache_file, sic_vals)
        with open(cache_meta, "w") as f:
            json.dump({"timestamp": datetime.utcnow().timestamp(), "shape": list(sic_vals.shape)}, f)

        print(f"[real-data] Successfully loaded real SIC data: {sic_vals.shape}")
        print(f"[real-data] Mean SIC: {sic_vals.mean():.1f}%, Max: {sic_vals.max():.1f}%")
        return sic_vals

    except Exception as e:
        print(f"[real-data] Failed to fetch real data: {e}")
        print("[real-data] Falling back to synthetic data")
        return None


# ---------------------------------------------------------------------------
# Synthetic data generators (fallback)
# ---------------------------------------------------------------------------

def generate_sic_history(n_days=15, seed=7):
    """
    Generate SIC history for the region.

    Tries real Copernicus Marine data first; falls back to synthetic if
    credentials aren't set up or the network is unavailable.
    """
    # Try real data first
    real_data = _fetch_real_sic(n_days=n_days)
    if real_data is not None:
        return real_data

    # Fallback: synthetic SIC history
    print("[synthetic] Generating synthetic SIC data")
    n = REGION["grid_size"]
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    coast_dist = np.sqrt((y - 4) ** 2 + (x - n * 0.3) ** 2)

    data = np.zeros((n_days, n, n), dtype=np.float32)
    for t in range(n_days):
        ice_edge_radius = 14 + t * 0.15  # slowly advancing ice edge
        base = 100 / (1 + np.exp((coast_dist - ice_edge_radius) * 0.5))
        noise = rng.normal(0, 2.5, size=base.shape)
        data[t] = np.clip(base + noise, 0, 100)
    return data


def generate_currents_and_wind(seed=7):
    """Synthetic constant-ish ocean current and wind fields for the region."""
    rng = np.random.default_rng(seed)
    # Weddell Gyre-like: westward/northward drift near the coast
    current_u = -0.10 + rng.normal(0, 0.02)
    current_v = 0.06 + rng.normal(0, 0.02)
    wind_u = 5.0 + rng.normal(0, 1.0)
    wind_v = -3.0 + rng.normal(0, 1.0)
    return {"current_u": current_u, "current_v": current_v, "wind_u": wind_u, "wind_v": wind_v}


if __name__ == "__main__":
    sic = generate_sic_history()
    env = generate_currents_and_wind()
    print("SIC history shape:", sic.shape)
    print("Environment:", env)
    print("Port A grid cell:", lonlat_to_grid(PORT_A["lon"], PORT_A["lat"]))
    print("Port B grid cell:", lonlat_to_grid(PORT_B["lon"], PORT_B["lat"]))
