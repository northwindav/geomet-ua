from typing import Dict, Optional

import pandas as pd
import requests

from get_geomet_ua import get_geomet_profiles


def get_true_elevation_m(lat: float, lon: float, cfg: Dict) -> Optional[float]:
    if not cfg.get("enabled", False):
        return None

    url = cfg.get("url", "").format(lat=lat, lon=lon)
    timeout = int(cfg.get("timeout_seconds", 15))
    if not url:
        return None

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload.get("elevation"), list) and payload["elevation"]:
            return float(payload["elevation"][0])
        if payload.get("elevation") is not None:
            return float(payload["elevation"])
    except Exception:
        return None

    return None


def trim_below_true_elevation(df: pd.DataFrame, true_elev_m: Optional[float]) -> pd.DataFrame:
    if true_elev_m is None or "geopotential height_dm" not in df.columns:
        return df

    mask = (df["geopotential height_dm"].astype(float) * 10.0) >= float(true_elev_m)
    return df[mask].reset_index(drop=True)


def fetch_geomet_forecast(
    lat: float,
    lon: float,
    model: str,
    time_window_h: int,
    time_step_h: int,
    elevation_cfg: Dict,
) -> pd.DataFrame:
    df = get_geomet_profiles(
        lat=lat,
        lon=lon,
        model=model,
        time_window_h=time_window_h,
        time_step_h=time_step_h,
    )
    true_elev_m = get_true_elevation_m(lat, lon, elevation_cfg)
    return trim_below_true_elevation(df, true_elev_m)
