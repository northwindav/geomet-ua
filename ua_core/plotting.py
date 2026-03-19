import os
from typing import Any, cast
from typing import Dict

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
from metpy.plots import Hodograph, SkewT, add_timestamp
from metpy.units import units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def make_title(site: str, skew_type: str, date: str, hour: str, fx_hour=None, zoom: bool = False, model=None) -> str:
    if skew_type == "obs":
        if zoom:
            return f"{site}_{skew_type}_{date}_{hour}UTC_skewT_loweratmos.png"
        return f"{site}_{skew_type}_{date}_{hour}UTC_skewT.png"

    model_prefix = f"{model}_" if model else ""
    if zoom:
        return f"{site}_{model_prefix}{skew_type}_{date}_{hour}UTCp{fx_hour}_skewT_loweratmos.png"
    return f"{site}_{model_prefix}{skew_type}_{date}_{hour}UTCp{fx_hour}_skewT.png"


def resolve_site_prefix(stn_id, stn_iata_code, stn_wmo_id, stn_lat, stn_lon):
    if stn_id:
        return str(stn_id)
    if stn_iata_code:
        return str(stn_iata_code)
    if stn_lat is not None and stn_lon is not None:
        lat_str = f"{stn_lat:.1f}".replace(".", "").replace("-", "")
        lon_str = f"{stn_lon:.1f}".replace(".", "").replace("-", "")
        return f"{lat_str}{lon_str}"
    return "site"


def plot_skewt(df: pd.DataFrame, zoom: bool = False):
    height_dm = df["geopotential height_dm"].astype(float)
    df = df[height_dm > 0.0].reset_index(drop=True)

    if zoom:
        df = df[df["pressure_hPa"] >= 500].reset_index(drop=True)
        ylim_bottom, ylim_top, aspect, rotation, hodo = 1000, 500, 250, 30, False
    else:
        ylim_bottom, ylim_top, aspect, rotation, hodo = 1000, 100, 80.5, 45, True

    pres = df["pressure_hPa"].values * units.hPa
    temp = df["temperature_C"].values * units.degC
    dewpoint = df["dew point temperature_C"].values * units.degC

    wind_speed = df["wind speed_kmh"].values * units.km / units.h
    wind_dir = df["wind direction_degree"].values * units.degrees
    u, v = mpcalc.wind_components(wind_speed, wind_dir)

    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, rotation=rotation, aspect=aspect)

    skew.plot(pres, temp, "red", label="Temperature")
    skew.plot(pres, dewpoint, "green", label="Dew Point")

    barb_interval = np.arange(150, 1000, 50) * units("hPa")
    ix = mpcalc.resample_nn_1d(pres, barb_interval)
    skew.plot_barbs(pres[ix], u[ix], v[ix], xloc=1)

    surface_idx = 0
    lcl_pressure, lcl_temperature = mpcalc.lcl(pres[surface_idx], temp[surface_idx], dewpoint[surface_idx])
    skew.plot(lcl_pressure, lcl_temperature, "ko", markerfacecolor="black", label="LCL")

    profile = mpcalc.parcel_profile(pres, temp[surface_idx], dewpoint[surface_idx]).to("degC")
    skew.plot(pres, profile, "k", linestyle="dashed", linewidth=2, label="Parcel Profile")

    skew.ax.set_xlabel("Temperature (C)")
    skew.ax.set_ylabel("Pressure (hPa)")
    skew.ax.set_ylim(ylim_bottom, ylim_top)
    skew.ax.set_xlim(-55, 30)

    skew.ax.axvline(0, color="c", linestyle="--", linewidth=2, label="0C Isotherm")
    skew.plot_dry_adiabats(t0=np.arange(200, 533, 15) * units.K, linewidth=1, alpha=0.3, color="orangered")
    skew.plot_moist_adiabats(linewidth=1, alpha=0.3, color="green")
    skew.plot_mixing_lines(pressure=np.arange(1000, 99, -25) * units.hPa, linewidth=1, linestyle="dotted", color="tab:blue")

    target_pressures = [1000, 900, 850, 800, 750, 700, 500, 300, 200, 100]
    pres_df = np.asarray(df["pressure_hPa"].dropna(), dtype=float)
    height_df = np.asarray(df["geopotential height_dm"].dropna(), dtype=float)
    if len(pres_df) > 1 and len(height_df) > 1:
        for p in target_pressures:
            if pres_df.min() <= p <= pres_df.max():
                h = np.interp(p, pres_df[::-1], height_df[::-1])
                skew.ax.text(-50, p, f"{h:.0f} dm", fontsize=9, color="gray", ha="left", va="center")

    if hodo:
        mask = height_df < 1200 # Plotting cutoff height in dm, not in m.
        uu = u[mask]
        vv = v[mask]
        height_df_12km = height_df[mask]

        ax_hod = inset_axes(skew.ax, "35%", "35%", loc="upper right")
        h = Hodograph(ax_hod, component_range=80)
        h.add_grid(increment=20, ls="-", lw=1.0, alpha=0.7)
        h.add_grid(increment=10, ls="--", lw=1, alpha=0.3)
        h.ax.set_box_aspect(1)
        h.ax.set_yticklabels([])
        h.ax.set_xticklabels([])
        h.ax.set_xticks([])
        h.ax.set_yticks([])
        h.ax.set_xlabel("")
        h.ax.set_ylabel("")

        for i in range(20, 160, 20):
            h.ax.annotate(str(i), (i, 0), xytext=(0, 2), textcoords="offset pixels", clip_on=True, fontsize=8, alpha=0.5, zorder=0)
        for i in range(20, 160, 20):
            h.ax.annotate(str(i), (0, i), xytext=(0, 2), textcoords="offset pixels", clip_on=True, fontsize=8, alpha=0.5, zorder=0)

        h.plot_colormapped(uu, vv, c=height_df_12km, cmap="viridis", label="0-12km wind")

    skew.ax.legend(loc="upper left")
    return skew


def save_single_plot(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    filename: str,
    figures_dir: str,
    zoom: bool = False,
):
    skewt = plot_skewt(df, zoom=zoom)
    skewt.ax.set_title(title, fontsize=11, pad=24, loc="center")
    skewt.ax.text(0.5, 1.02, subtitle, fontsize="medium", ha="center", va="bottom", transform=skewt.ax.transAxes)

    current_utc = pd.Timestamp.utcnow()
    add_timestamp(skewt.ax, time=current_utc, y=-0.10, x=0.0, ha="left", time_format="%Y-%m-%d %H:%M UTC", fontsize="medium")
    skewt.ax.text(1.08, 0.5, "Wind (km/h)", transform=skewt.ax.transAxes, rotation=90, va="center", ha="left", fontsize="medium")

    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    fig = cast(Any, skewt.ax.figure)
    fig.savefig(filepath)


def save_forecast_plots(
    df: pd.DataFrame,
    model_time: pd.Timestamp,
    model: str,
    station_meta: Dict,
    plot_config: Dict,
    figures_dir: str,
):
    site_prefix = resolve_site_prefix(
        plot_config.get("stn_id"),
        station_meta.get("stn_iata_code"),
        station_meta.get("stn_wmo_id"),
        station_meta.get("stn_lat"),
        station_meta.get("stn_lon"),
    )

    station_elev_m = station_meta.get("stn_elev")
    if station_elev_m is not None:
        try:
            station_elev_m = int(round(float(station_elev_m)))
        except Exception:
            station_elev_m = None

    csv_model_elev_m = station_meta.get("stn_mod_elev")
    if csv_model_elev_m is not None:
        try:
            csv_model_elev_m = int(round(float(csv_model_elev_m)))
        except Exception:
            csv_model_elev_m = None

    geomet_model_elev_m = None
    if "model_surface_elev_dm" in df.columns:
        model_elev_series = df["model_surface_elev_dm"].dropna()
        if not model_elev_series.empty:
            geomet_model_elev_m = int(round(float(model_elev_series.iloc[0]) * 10.0))

    source_tag = plot_config.get("forecast_source")
    source_text = str(source_tag).strip() if source_tag else "unknown"

    for fx_hour in sorted(df["forecast_hour"].unique()):
        df_fh = df[df["forecast_hour"] == fx_hour]
        title = (
            f"{model} Forecast for {station_meta['stn_name']}: Model init {plot_config['date']} "
            f"{plot_config['hour']}UTC. Valid {model_time + pd.Timedelta(hours=fx_hour)}UTC (+{fx_hour}h)"
        )

        model_elev_m = None
        if "model_surface_elev_dm" in df_fh.columns and df_fh["model_surface_elev_dm"].notna().any():
            model_elev_m = int(round(float(df_fh["model_surface_elev_dm"].dropna().iloc[0]) * 10.0))
        elif geomet_model_elev_m is not None:
            model_elev_m = geomet_model_elev_m
        elif csv_model_elev_m is not None and csv_model_elev_m > 0:
            model_elev_m = csv_model_elev_m

        if model_elev_m is not None and model_elev_m < 0 and station_elev_m is not None and station_elev_m > 0:
            model_elev_m = station_elev_m + model_elev_m

        station_elev_str = f"{station_elev_m} m" if station_elev_m is not None and station_elev_m > 0 else "n/a"
        model_elev_str = f"{model_elev_m} m" if model_elev_m is not None else "n/a"

        subtitle = (
            f"{station_meta['stn_iata_code']}/{station_meta['stn_wmo_id']}, "
            f"Lat: {station_meta['stn_lat']:.2f}, Lon: {station_meta['stn_lon']:.2f}, "
            f"Station elev: {station_elev_str}, Model elev: {model_elev_str}, Source: {source_text}"
        )

        filename = make_title(
            site_prefix,
            plot_config["skew_type"],
            plot_config["date"],
            plot_config["hour"],
            fx_hour,
            zoom=plot_config.get("zoom", False),
            model=model,
        )
        save_single_plot(df_fh, title, subtitle, filename, figures_dir, zoom=plot_config.get("zoom", False))
