import pandas as pd


def build_datamart_url(date: str, hour: str, station_id: str, template: str) -> str:
    return template.format(date=date, hour=hour, station=station_id.upper())


def reshape_eccc_df(df_raw: pd.DataFrame, fx_hours=None) -> pd.DataFrame:
    if fx_hours is None:
        fx_hours = [0, 6, 12, 18, 24, 36, 48]

    df_raw = df_raw.rename(
        columns={
            "Variable": "variable",
            "Level": "pressure",
            "Fcst_Hr": "forecast_hour",
            "Value": "value",
        }
    )
    df_raw = df_raw.sort_values(["forecast_hour", "pressure"], ascending=[True, False])
    df_raw = df_raw[df_raw["pressure"].between(100, 1015)]

    dfs = {}
    for fh in sorted(df_raw["forecast_hour"].unique()):
        df_fh = df_raw[df_raw["forecast_hour"] == fh]
        df_pivot = df_fh.pivot(index="pressure", columns="variable", values="value").reset_index()
        df_pivot = df_pivot.sort_values("pressure")

        rename_dict = {
            "pressure": "pressure_hPa",
            "TT": "temperature_C",
            "WD": "wind direction_degree",
            "UV": "wind speed_kt",
            "ES": "dewpoint_depression_C",
            "GZ": "geopotential height_dm",
        }
        df_pivot = df_pivot.rename(columns=rename_dict)

        if "temperature_C" in df_pivot.columns and "dewpoint_depression_C" in df_pivot.columns:
            df_pivot["dew point temperature_C"] = (
                df_pivot["temperature_C"] - df_pivot["dewpoint_depression_C"]
            )

        dfs[fh] = df_pivot

    all_dfs = []
    for fh, df_pivot in dfs.items():
        df_pivot["forecast_hour"] = fh
        all_dfs.append(df_pivot)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.sort_values(["forecast_hour", "pressure_hPa"], ascending=[True, False])

    df_all["wind speed_kmh"] = df_all["wind speed_kt"].astype(float) * 1.852
    df_all["geopotential height_dm"] = df_all["geopotential height_dm"].astype(float) * 10

    df_all = df_all[df_all["forecast_hour"].isin(fx_hours)]

    subset = ["pressure_hPa", "temperature_C", "dew point temperature_C"]
    if "relative humidity_%" in df_all.columns:
        subset.append("relative humidity_%")
    df_all = df_all.dropna(subset=subset)

    return df_all.reset_index(drop=True)


def fetch_datamart_forecast(date: str, hour: str, station_id: str, url_template: str) -> pd.DataFrame:
    url = build_datamart_url(date, hour, station_id, url_template)
    df_raw = pd.read_csv(url, header=1, skiprows=[2])
    return reshape_eccc_df(df_raw)
