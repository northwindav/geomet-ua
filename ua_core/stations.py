from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass
class StationMetadata:
    stn_lat: float
    stn_lon: float
    stn_elev: int
    stn_mod_elev: int
    stn_name: str
    stn_upperair_obs: bool
    stn_iata_code: str
    stn_wmo_id: Optional[int]


def load_stations(station_file: str) -> pd.DataFrame:
    return pd.read_csv(station_file)


def find_station_by_id(stn_id: str, stations: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    stn_id_str = str(stn_id).strip()
    if not stn_id_str:
        return None, None

    if stn_id_str.isdigit():
        matched = stations[stations["wmo_id"].astype("Int32") == int(stn_id_str)]
        match_type = "WMO ID"
    else:
        matched = stations[stations["iata_code"].str.upper() == stn_id_str.upper()]
        match_type = "IATA/ICAO code"

    if not matched.empty:
        return matched.iloc[0], match_type
    return None, match_type


def find_station_by_latlon(lat: float, lon: float, stations: pd.DataFrame) -> Optional[pd.Series]:
    lat_rounded = round(lat, 1)
    lon_rounded = round(lon, 1)

    matched = stations[
        (stations["lat"].round(1) == lat_rounded)
        & (stations["lon"].round(1) == lon_rounded)
    ]

    if not matched.empty:
        return matched.iloc[0]
    return None


def extract_station_metadata(station_row: pd.Series) -> StationMetadata:
    return StationMetadata(
        stn_lat=float(station_row["lat"]),
        stn_lon=float(station_row["lon"]),
        stn_elev=int(station_row["elev_m"]),
        stn_mod_elev=int(station_row["rdps_elev_m"]),
        stn_name=str(station_row["name"]),
        stn_upperair_obs=bool(station_row["upperair_obs"]),
        stn_iata_code=str(station_row["iata_code"]),
        stn_wmo_id=int(station_row["wmo_id"]) if pd.notna(station_row["wmo_id"]) else None,
    )


def as_dict(meta: StationMetadata) -> dict:
    return {
        "stn_lat": meta.stn_lat,
        "stn_lon": meta.stn_lon,
        "stn_elev": meta.stn_elev,
        "stn_mod_elev": meta.stn_mod_elev,
        "stn_name": meta.stn_name,
        "stn_upperair_obs": meta.stn_upperair_obs,
        "stn_iata_code": meta.stn_iata_code,
        "stn_wmo_id": meta.stn_wmo_id,
    }
