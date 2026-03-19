import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

from ua_core.bufr import try_fetch_bufr_observed_profile
from ua_core.datamart import fetch_datamart_forecast
from ua_core.geomet import fetch_geomet_forecast, get_true_elevation_m
from ua_core.plotting import make_title, resolve_site_prefix, save_forecast_plots, save_single_plot
from ua_core.stations import as_dict, extract_station_metadata, find_station_by_id, find_station_by_latlon, load_stations
from ua_core.wyoming import fetch_wyoming_observed


MODEL_BOUNDS = {
    "HRDPS": {"lat_min": 37.5, "lat_max": 68.0, "lon_min": -142.0, "lon_max": -45.0},
    "RDPS": {"lat_min": 30.0, "lat_max": 75.0, "lon_min": -155.0, "lon_max": -35.0},
    "GDPS": {"lat_min": -90.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0},
}


@dataclass
class RequestOptions:
    stn_id: str
    date: str
    hour: str
    skew_type: str
    zoom: bool
    logfile: Optional[str]
    location_mode: str
    lat: Optional[float]
    lon: Optional[float]
    model: str
    input_file: Optional[str]
    validate_only: bool = False


def log_message(msg: str, logfile: Optional[str] = None):
    print(msg)
    if logfile:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")


def load_user_profile(file_path: str, skew_type: str, logfile: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(file_path):
        err = f"Input file not found: {file_path}"
        log_message(err, logfile)
        raise ValueError(err)

    df = pd.read_csv(file_path)

    base_cols = [
        "pressure_hPa",
        "temperature_C",
        "dew point temperature_C",
        "wind direction_degree",
        "wind speed_kmh",
        "geopotential height_dm",
    ]
    required = list(base_cols)
    if skew_type.lower() == "fx":
        required.append("forecast_hour")

    missing = [c for c in required if c not in df.columns]
    if missing:
        err = f"Input file missing required columns: {', '.join(missing)}"
        log_message(err, logfile)
        raise ValueError(err)

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)

    if "forecast_hour" in df.columns:
        df["forecast_hour"] = df["forecast_hour"].astype(int)

    if "forecast_hour" in df.columns:
        df = df.sort_values(["forecast_hour", "pressure_hPa"], ascending=[True, False])
    else:
        df = df.sort_values("pressure_hPa", ascending=False)

    log_message(f"Loaded user profile from {file_path} with {len(df)} rows.", logfile)
    return df.reset_index(drop=True)


def _resolve_location(opts: RequestOptions, cfg: Dict) -> Dict:
    station_file = cfg["paths"]["station_file"]
    stations = load_stations(station_file)

    stn_meta = {
        "stn_lat": None,
        "stn_lon": None,
        "stn_elev": 0,
        "stn_mod_elev": 0,
        "stn_name": "Unknown",
        "stn_upperair_obs": False,
        "stn_iata_code": "UNKN",
        "stn_wmo_id": None,
    }
    station_found = False

    if opts.location_mode == "file":
        stn_meta.update(
            {
                "stn_lat": 0.0,
                "stn_lon": 0.0,
                "stn_name": "User file",
                "stn_iata_code": "FILE",
                "stn_wmo_id": 0,
            }
        )
        log_message("Using user-provided sounding file; skipping station lookup.", opts.logfile)
        return {"station_meta": stn_meta, "station_found": False}

    if opts.location_mode == "station":
        station_row, _ = find_station_by_id(opts.stn_id, stations)
        if station_row is not None:
            stn_meta = as_dict(extract_station_metadata(station_row))
            station_found = True
            log_message(
                f"Found station {stn_meta['stn_iata_code']}/{stn_meta['stn_wmo_id']} ({stn_meta['stn_name']}) at "
                f"{stn_meta['stn_lat']:.2f}, {stn_meta['stn_lon']:.2f}",
                opts.logfile,
            )
        elif opts.lat is not None and opts.lon is not None:
            stn_meta.update(
                {
                    "stn_lat": opts.lat,
                    "stn_lon": opts.lon,
                    "stn_name": f"Point {opts.lat:.2f}, {opts.lon:.2f}",
                    "stn_iata_code": "UNKN",
                    "stn_wmo_id": 99999,
                }
            )
            log_message(
                f"Station ID {opts.stn_id} not found in station file. Using provided lat/lon: {opts.lat:.2f}, {opts.lon:.2f}",
                opts.logfile,
            )
        else:
            raise ValueError(
                f"Station ID {opts.stn_id} not found in station file and no lat/lon provided."
            )
    else:
        if opts.lat is None or opts.lon is None:
            raise ValueError("Lat/lon mode requires both --lat and --lon.")

        station_row = find_station_by_latlon(opts.lat, opts.lon, stations)
        if station_row is not None:
            stn_meta = as_dict(extract_station_metadata(station_row))
            station_found = True
            log_message(
                f"Matched lat/lon to station {stn_meta['stn_iata_code']}/{stn_meta['stn_wmo_id']} ({stn_meta['stn_name']})",
                opts.logfile,
            )
        else:
            stn_meta.update(
                {
                    "stn_lat": opts.lat,
                    "stn_lon": opts.lon,
                    "stn_name": f"Point {opts.lat:.2f}, {opts.lon:.2f}",
                    "stn_iata_code": "UNKN",
                    "stn_wmo_id": 99999,
                }
            )
            log_message(
                f"No matching station for lat/lon {opts.lat:.2f}, {opts.lon:.2f}. Using point location.",
                opts.logfile,
            )

    return {"station_meta": stn_meta, "station_found": station_found}


def _validate_request(opts: RequestOptions):
    if opts.skew_type not in ["obs", "fx"]:
        raise ValueError("skew_type must be obs or fx")

    if opts.hour not in ["00", "06", "12", "18"]:
        raise ValueError("hour must be one of 00, 06, 12, 18")

    date_test = pd.to_datetime(opts.date, format="%Y%m%d", utc=True)
    if not (date_test <= pd.Timestamp.utcnow().normalize()):
        raise ValueError("date must be in the past or present (UTC)")
    if opts.skew_type == "fx" and (date_test + pd.Timedelta(days=30)) < pd.Timestamp.utcnow().normalize():
        raise ValueError("Forecast soundings are only available from Datamart for the past 30 days")

    if opts.skew_type == "obs" and opts.location_mode == "latlon":
        raise ValueError("Observed soundings currently require station-id mode")


def _validate_domain(opts: RequestOptions, stn_meta: Dict):
    if opts.skew_type != "fx" or opts.location_mode == "file":
        return

    bounds = MODEL_BOUNDS.get(opts.model)
    if not bounds:
        return

    stn_lat = stn_meta.get("stn_lat")
    stn_lon = stn_meta.get("stn_lon")
    if stn_lat is None or stn_lon is None:
        return

    if not (
        bounds["lat_min"] <= stn_lat <= bounds["lat_max"]
        and bounds["lon_min"] <= stn_lon <= bounds["lon_max"]
    ):
        raise ValueError(
            f"Point ({stn_lat:.2f}, {stn_lon:.2f}) is outside {opts.model} domain "
            f"lat {bounds['lat_min']:.1f}-{bounds['lat_max']:.1f}, "
            f"lon {bounds['lon_min']:.1f}-{bounds['lon_max']:.1f}"
        )


def run_request(opts: RequestOptions, cfg: Dict):
    _validate_request(opts)

    location = _resolve_location(opts, cfg)
    station_meta = location["station_meta"]
    station_found = location["station_found"]

    _validate_domain(opts, station_meta)

    if opts.validate_only:
        log_message(
            (
                "Validation-only mode: request parsed and validated "
                f"(skew_type={opts.skew_type}, model={opts.model}, location_mode={opts.location_mode})."
            ),
            opts.logfile,
        )
        return

    figures_dir = cfg["paths"]["figures_dir"]
    plot_config = {
        "skew_type": opts.skew_type,
        "date": opts.date,
        "hour": opts.hour,
        "zoom": opts.zoom,
        "logfile": opts.logfile,
        "stn_id": opts.stn_id,
        "forecast_source": None,
    }

    if opts.input_file:
        df = load_user_profile(opts.input_file, opts.skew_type, opts.logfile)
        if opts.skew_type == "fx":
            plot_config["forecast_source"] = "User file"
            model_time = pd.to_datetime(f"{opts.date} {opts.hour}", format="%Y%m%d %H")
            save_forecast_plots(df, model_time, opts.model, station_meta, plot_config, figures_dir)
            return

        title = f"User Skew-T valid {opts.date} {opts.hour}UTC"
        subtitle = "User-provided profile"
        site_prefix = resolve_site_prefix(
            opts.stn_id,
            station_meta.get("stn_iata_code"),
            station_meta.get("stn_wmo_id"),
            station_meta.get("stn_lat"),
            station_meta.get("stn_lon"),
        )
        filename = make_title(site_prefix, opts.skew_type, opts.date, opts.hour, zoom=opts.zoom)
        save_single_plot(df, title, subtitle, filename, figures_dir, zoom=opts.zoom)
        log_message(f"Saved figure: {filename}", opts.logfile)
        return

    if opts.skew_type == "obs":
        df_bufr = None
        obs_source = "unknown"
        if opts.location_mode == "station":
            df_bufr, bufr_meta, bufr_filename = try_fetch_bufr_observed_profile(
                opts.date,
                opts.hour,
                opts.stn_id,
                cfg.get("bufr", {}),
            )
            if df_bufr is not None and not df_bufr.empty:
                obs_source = "BUFR (Datamart)"
                log_message(f"Retrieved observed profile from BUFR file: {bufr_filename}", opts.logfile)
                if bufr_meta is not None:
                    if bufr_meta.get("bufr_bulletin"):
                        log_message(
                            f"BUFR bulletin selected: {bufr_meta['bufr_bulletin']} ({bufr_meta.get('bufr_url', 'n/a')})",
                            opts.logfile,
                        )
                    for key in ["stn_lat", "stn_lon", "stn_elev"]:
                        if bufr_meta.get(key) is not None:
                            station_meta[key] = bufr_meta[key]
            elif cfg.get("bufr", {}).get("enabled", False):
                log_message("[obs] BUFR unavailable; using Wyoming CSV fallback.", opts.logfile)

        if df_bufr is None:
            if station_meta.get("stn_wmo_id") is None:
                raise ValueError("Observed fallback requires a valid WMO station ID in station metadata.")

            df_bufr = fetch_wyoming_observed(
                opts.date,
                opts.hour,
                station_meta["stn_wmo_id"],
                cfg["endpoints"]["wyoming_csv"],
            )
            obs_source = "U. Wyoming"
            log_message("[obs] Retrieved profile from Wyoming CSV fallback.", opts.logfile)

        title = f"Observed Skew-T for {station_meta['stn_name']} valid {opts.date} {opts.hour}UTC"
        subtitle = (
            f"{station_meta['stn_iata_code']}/{station_meta['stn_wmo_id']}, "
            f"Lat: {station_meta['stn_lat']:.2f}, Lon: {station_meta['stn_lon']:.2f}, "
            f"Elev: {station_meta['stn_elev']} m, Source: {obs_source}"
        )
        site_prefix = resolve_site_prefix(
            opts.stn_id,
            station_meta.get("stn_iata_code"),
            station_meta.get("stn_wmo_id"),
            station_meta.get("stn_lat"),
            station_meta.get("stn_lon"),
        )
        filename = make_title(site_prefix, "obs", opts.date, opts.hour, zoom=opts.zoom)
        save_single_plot(df_bufr, title, subtitle, filename, figures_dir, zoom=opts.zoom)
        log_message(f"Saved figure: {filename}", opts.logfile)
        return

    # For forecast plots, ensure station elevation is available for subtitle metadata.
    station_elev = station_meta.get("stn_elev")
    needs_station_elev = (
        station_elev is None
        or pd.isna(station_elev)
        or (isinstance(station_elev, (int, float)) and float(station_elev) <= 0)
    )
    station_elev_api = None
    if needs_station_elev:
        stn_lat = station_meta.get("stn_lat")
        stn_lon = station_meta.get("stn_lon")
        if stn_lat is not None and stn_lon is not None:
            station_elev_api = get_true_elevation_m(
                float(stn_lat),
                float(stn_lon),
                cfg.get("elevation_api", {}),
            )
            if station_elev_api is not None:
                station_meta["stn_elev"] = int(round(float(station_elev_api)))
                log_message(
                    f"[fx] Station elevation resolved via API: {station_meta['stn_elev']} m",
                    opts.logfile,
                )

    # RDPS model elevation normally comes from station_data.csv (rdps_elev_m).
    # If that value is missing, use elevation API as a fallback so all forecast plots can show model elev.
    if opts.model == "RDPS":
        model_elev = station_meta.get("stn_mod_elev")
        needs_model_elev = (
            model_elev is None
            or pd.isna(model_elev)
            or (isinstance(model_elev, (int, float)) and float(model_elev) <= 0)
        )
        if needs_model_elev:
            if station_elev_api is None:
                stn_lat = station_meta.get("stn_lat")
                stn_lon = station_meta.get("stn_lon")
                if stn_lat is not None and stn_lon is not None:
                    station_elev_api = get_true_elevation_m(
                        float(stn_lat),
                        float(stn_lon),
                        cfg.get("elevation_api", {}),
                    )
            if station_elev_api is not None:
                station_meta["stn_mod_elev"] = int(round(float(station_elev_api)))
                log_message(
                    f"[fx][RDPS] Model elevation resolved via API: {station_meta['stn_mod_elev']} m",
                    opts.logfile,
                )

    if opts.model == "RDPS" and station_found:
        try:
            df = fetch_datamart_forecast(
                opts.date,
                opts.hour,
                station_meta["stn_iata_code"],
                cfg["endpoints"]["datamart_vertical_profile"],
            )
            if df is not None and not df.empty:
                plot_config["forecast_source"] = "Datamart"
                model_time = pd.to_datetime(f"{opts.date} {opts.hour}", format="%Y%m%d %H")
                save_forecast_plots(df, model_time, opts.model, station_meta, plot_config, figures_dir)
                return

            log_message(
                "[fx][RDPS] Datamart returned no rows; using GeoMet fallback.",
                opts.logfile,
            )
        except Exception as exc:
            log_message(
                f"[fx][RDPS] Datamart retrieval failed ({exc}); using GeoMet fallback.",
                opts.logfile,
            )

    df = fetch_geomet_forecast(
        station_meta["stn_lat"],
        station_meta["stn_lon"],
        opts.model,
        int(cfg["defaults"].get("time_window_h", 48)),
        int(cfg["defaults"].get("time_step_h", 3)),
        cfg.get("elevation_api", {}),
    )
    if df is None or df.empty:
        raise SystemExit("GeoMet returned no data for the requested location/time window")

    if opts.model == "RDPS" and station_found:
        plot_config["forecast_source"] = "GeoMet (RDPS fallback)"
    else:
        plot_config["forecast_source"] = "GeoMet"

    model_time = pd.to_datetime(f"{opts.date} {opts.hour}", format="%Y%m%d %H")
    save_forecast_plots(df, model_time, opts.model, station_meta, plot_config, figures_dir)


def init_log(logfile: str):
    log_dir = os.path.dirname(logfile)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    content = [
        "==========================================",
        f"Start time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "-----Begin plot_ua.py output-----",
    ]
    with open(logfile, "w", encoding="utf-8") as f:
        f.write("\n".join(content) + "\n")


def close_log(logfile: str):
    with open(logfile, "a", encoding="utf-8") as f:
        f.write("-----End plot_ua.py output-----\n")
        f.write(f"Script complete at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}.\n")
