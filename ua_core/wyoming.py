from typing import Optional

import pandas as pd


def build_wyoming_url(date: str, hour: str, station_id: str, template: str) -> str:
    date_formatted = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
    return template.format(date=date_formatted, hour=hour, station=station_id)


def reshape_wyoming_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = df_raw.rename(columns={"geopotential height_m": "geopotential height_dm"})

    key_cols = [
        "pressure_hPa",
        "temperature_C",
        "dew point temperature_C",
        "wind speed_m/s",
        "wind direction_degree",
    ]
    for col in key_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df_raw["wind speed_kmh"] = df_raw["wind speed_m/s"].astype(float) * 3.6

    df_raw = df_raw.dropna(
        subset=[
            "pressure_hPa",
            "temperature_C",
            "dew point temperature_C",
            "relative humidity_%",
        ]
    )
    return df_raw.reset_index(drop=True)


def fetch_wyoming_observed(
    date: str,
    hour: str,
    station_wmo_id: int,
    url_template: str,
    timeout: Optional[int] = None,
) -> pd.DataFrame:
    url = build_wyoming_url(date, hour, str(station_wmo_id), url_template)
    # pandas uses urllib; timeout is not directly exposed in this path.
    df_raw = pd.read_csv(url, sep=",", header=0)
    return reshape_wyoming_df(df_raw)
