#!/usr/bin/env python3
# Plot ua
# Use kernel geomet-ua 
# Smith Dec 2025. Michael.Smith2@nrcan-rncan.gc.ca
# Retrieves and plots observed/fx soundings from UW and ECCC
# We don't use the ECCC observed UA because of missing data issues
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.plots import SkewT, add_timestamp, Hodograph
from metpy.units import units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime, timezone
import argparse
from get_geomet_ua import get_geomet_profiles


# To-do:
# Add an additional check for when observed soundings are requested, and return an error if the station ID is not valid for UW data
# Add summer option that will:
#   Add shading for CAPE and CIN
#   Add calculations to the mix and figure out where to plot them. See https://projectpythia.org/metpy-cookbook/notebooks/skewt/sounding-calculations/
# For some other layout and examples see https://unidata.github.io/MetPy/latest/examples/Advanced_Sounding_With_Complex_Layout.html#sphx-glr-examples-advanced-sounding-with-complex-layout-py 
# Thread-safety: If parallelizing model calls, ensure get_geomet_ua.py is refactored to avoid global state mutation.
#   Consider passing context (point, time window, cache) as parameters or use instance-based caching.


# =============================================================================
# User options
# =============================================================================
# List of stations for fx: https://dd.weather.gc.ca/20251223/WXO-DD/vertical_profile/doc/station_list_for_vertical_profile.txt
# List of stations for obs can be found at: https://weather.uwyo.edu/upperair/sounding_legacy.html. 

# Get today's date in UTC as default
today_utc = datetime.now(timezone.utc).strftime('%Y%m%d')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot upper air soundings from UW or ECCC')
parser.add_argument('--stn_id', type=str, default='cyxy', help='Station ID (WMO or ICAO code)')
parser.add_argument('--date', type=str, default=today_utc, help='Date in YYYYMMDD format')
parser.add_argument('--hour', type=str, default='00', choices=['00', '06', '12', '18'], help='Observation or model init hour, in UTC (00, 06, 12, or 18)')
parser.add_argument('--skew_type', type=str, default='obs', choices=['obs', 'fx'], help='Type of sounding: obs or fx')
parser.add_argument('--zoom', action='store_true', help='Zoom to lower atmosphere')
parser.add_argument('--logfile', type=str, default=None, help='Logfile path for output messages')
parser.add_argument('--location_mode', type=str, default='station', choices=['station', 'latlon', 'file'], help='Location mode: station ID, latlon, or file input')
parser.add_argument('--lat', type=float, default=None, help='Latitude, decimal degrees (used in latlon mode)')
parser.add_argument('--lon', type=float, default=None, help='Longitude, decimal degrees (used in latlon mode)')
parser.add_argument('--model', type=str, default='HRDPS', choices=['HRDPS', 'RDPS', 'GDPS'], help='Model selection for forecast (HRDPS, RDPS, or GDPS)')
parser.add_argument('--input_file', type=str, default=None, help='Path to CSV containing sounding data (bypasses data retrieval)')

args = parser.parse_args()

stn_id = args.stn_id
date = args.date
hour = args.hour
skew_type = args.skew_type
zoom = args.zoom
logfile = args.logfile
location_mode = args.location_mode
lat = args.lat
lon = args.lon
model = args.model.upper()
input_file = args.input_file

# ----- you normally don't need to change anything below this line -----

# Check on inputs and exit if invalid
if skew_type.lower() not in ['obs', 'fx']:
    raise ValueError("skew_type must be either 'obs' or 'fx'")

if hour not in ['00','06', '12', '18']:
    raise ValueError("hour must be either '00', '06', '12' or '18'")

date_test = pd.to_datetime(date, format='%Y%m%d', utc=True)  # will raise error if invalid
if not (date_test <= pd.Timestamp.utcnow().normalize()):
    raise ValueError("date must be in the past or present (UTC)")
elif skew_type.lower() == 'fx' and (date_test + pd.Timedelta(days=30)) < (pd.Timestamp.utcnow().normalize() ):
    raise ValueError("Forecast soundings are only available from Datamart for the past 30 days")


# ---- Logic to retrieve station data: --------
# If station_mode: Check for station in station_data.csv. If present, grab the lat/lon/elev info for plotting. Then, if model is RDPS we'll retrieve from datamart, otherwise from geomet.
# If latlon mode: use the provided lat/lon and skip station data lookup. Note that observed soundings are only available via station ID.

# Helper function to write messages to both console and logfile
def log_message(msg, logfile=None):
    print(msg)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(f"{msg}\n")

# Helper to load and validate user-provided sounding CSV
def load_user_profile(file_path, skew_type, logfile=None):
    if not os.path.exists(file_path):
        err = f"Input file not found: {file_path}"
        log_message(err, logfile)
        raise ValueError(err)

    df = pd.read_csv(file_path)

    base_cols = [
        'pressure_hPa',
        'temperature_C',
        'dew point temperature_C',
        'wind direction_degree',
        'wind speed_kmh',
        'geopotential height_dm'
    ]
    required = base_cols.copy()
    if skew_type.lower() == 'fx':
        required.append('forecast_hour')

    missing = [c for c in required if c not in df.columns]
    if missing:
        err = f"Input file missing required columns: {', '.join(missing)}"
        log_message(err, logfile)
        raise ValueError(err)

    # Convert to numeric and drop rows with NaN in required fields
    numeric_cols = required
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=required)

    # Forecast hours as int if present
    if 'forecast_hour' in df.columns:
        df['forecast_hour'] = df['forecast_hour'].astype(int)

    # Sort for plotting
    if 'forecast_hour' in df.columns:
        df = df.sort_values(['forecast_hour', 'pressure_hPa'], ascending=[True, False])
    else:
        df = df.sort_values('pressure_hPa', ascending=False)

    log_message(f"Loaded user profile from {file_path} with {len(df)} rows.", logfile)
    return df.reset_index(drop=True)

# Helper function to find station by ID in station_data.csv
def find_station_by_id(stn_id, stations):
    stn_id_str = str(stn_id).strip()
    if not stn_id_str:  # Empty station ID
        return None, None
    
    if stn_id_str.isdigit():
        matched = stations[stations['wmo_id'].astype('Int32') == int(stn_id_str)]
        match_type = 'WMO ID'
    else:
        matched = stations[stations['iata_code'].str.upper() == stn_id_str.upper()]
        match_type = 'IATA/ICAO code'
    
    if not matched.empty:
        return matched.iloc[0], match_type
    return None, match_type

# Helper function to find station by lat/lon (to 1 decimal place)
def find_station_by_latlon(lat, lon, stations):
    lat_rounded = round(lat, 1)
    lon_rounded = round(lon, 1)
    
    matched = stations[
        (stations['lat'].round(1) == lat_rounded) & 
        (stations['lon'].round(1) == lon_rounded)
    ]
    
    if not matched.empty:
        return matched.iloc[0]
    return None

# Helper function to extract station metadata from a matched row
def extract_station_metadata(station_row):
    return {
        'stn_lat': float(station_row['lat']),
        'stn_lon': float(station_row['lon']),
        'stn_elev': int(station_row['elev_m']),
        'stn_mod_elev': int(station_row['rdps_elev_m']),
        'stn_name': str(station_row['name']),
        'stn_upperair_obs': bool(station_row['upperair_obs']),
        'stn_iata_code': str(station_row['iata_code']),
        'stn_wmo_id': int(station_row['wmo_id']) if pd.notna(station_row['wmo_id']) else None
    }

# Read station data
stations = pd.read_csv("station_data.csv")

# Initialize station metadata variables
stn_lat = None
stn_lon = None
stn_elev = None
stn_mod_elev = None
stn_name = None
stn_upperair_obs = None
stn_iata_code = None
stn_wmo_id = None
station_found = False

# === VALIDATE LAT/LON BOUNDS FOR FORECAST REQUESTS (early check) ===
# Define domain boundaries for each model (approximate, based on ECCC specifications)
# These bounds help catch invalid requests before data retrieval
MODEL_BOUNDS = {
    'HRDPS': {'lat_min': 37.5, 'lat_max': 68.0, 'lon_min': -142.0, 'lon_max': -45.0},  # Continental Canada
    'RDPS':  {'lat_min': 30.0, 'lat_max': 75.0, 'lon_min': -155.0, 'lon_max': -35.0},  # Canada + US domain
    'GDPS':  {'lat_min': -90.0, 'lat_max': 90.0, 'lon_min': -180.0, 'lon_max': 180.0}   # Global
}

# === LOCATION RESOLUTION LOGIC ===
if location_mode == 'file':
    # User-provided file; skip station lookup and data retrieval
    stn_lat = 0.0
    stn_lon = 0.0
    stn_elev = 0
    stn_mod_elev = 0
    stn_name = "User file"
    stn_upperair_obs = False
    stn_iata_code = "FILE"
    stn_wmo_id = 0
    station_found = False
    log_message("Using user-provided sounding file; skipping station lookup.", logfile)
elif location_mode == 'station':
    # Station mode: lookup by station ID
    station_row, match_type = find_station_by_id(stn_id, stations)
    
    if station_row is not None:
        metadata = extract_station_metadata(station_row)
        stn_lat = metadata['stn_lat']
        stn_lon = metadata['stn_lon']
        stn_elev = metadata['stn_elev']
        stn_mod_elev = metadata['stn_mod_elev']
        stn_name = metadata['stn_name']
        stn_upperair_obs = metadata['stn_upperair_obs']
        stn_iata_code = metadata['stn_iata_code']
        stn_wmo_id = metadata['stn_wmo_id']
        station_found = True
        log_message(f"Found station {stn_iata_code}/{stn_wmo_id} ({stn_name}) at {stn_lat:.2f}, {stn_lon:.2f}", logfile)
    else:
        # Station not found in CSV
        if lat is not None and lon is not None:
            # Fallback to provided lat/lon
            stn_lat = lat
            stn_lon = lon
            stn_elev = 0  # Unknown elevation
            stn_mod_elev = 0
            stn_name = f"Point {lat:.2f}N, {lon:.2f}E"
            stn_iata_code = "UNKN"
            stn_wmo_id = 99999
            log_message(f"Station ID {stn_id} not found in station_data.csv. Using provided lat/lon: {lat:.2f}, {lon:.2f}", logfile)
        else:
            # No fallback available
            err_msg = f"Station ID {stn_id} not found in station_data.csv and no lat/lon provided. Cannot proceed."
            log_message(err_msg, logfile)
            raise ValueError(err_msg)

else:  # location_mode == 'latlon'
    # Lat/lon mode: use provided coordinates, optionally match to station
    if lat is None or lon is None:
        err_msg = "Lat/lon mode requires both --lat and --lon parameters"
        log_message(err_msg, logfile)
        raise ValueError(err_msg)
    
    # Try to find matching station for metadata
    station_row = find_station_by_latlon(lat, lon, stations)
    
    if station_row is not None:
        metadata = extract_station_metadata(station_row)
        stn_lat = metadata['stn_lat']
        stn_lon = metadata['stn_lon']
        stn_elev = metadata['stn_elev']
        stn_mod_elev = metadata['stn_mod_elev']
        stn_name = metadata['stn_name']
        stn_upperair_obs = metadata['stn_upperair_obs']
        stn_iata_code = metadata['stn_iata_code']
        stn_wmo_id = metadata['stn_wmo_id']
        station_found = True
        log_message(f"Matched lat/lon to station {stn_iata_code}/{stn_wmo_id} ({stn_name})", logfile)
    else:
        # No matching station, use provided coordinates
        stn_lat = lat
        stn_lon = lon
        stn_elev = 0
        stn_mod_elev = 0
        stn_name = f"Point {lat:.2f}N, {lon:.2f}E"
        stn_iata_code = "UNKN"
        stn_wmo_id = 99999
        log_message(f"No matching station for lat/lon {lat:.2f}, {lon:.2f}. Using point location.", logfile)

# Validate lat/lon bounds against model domain if forecast is requested
if skew_type.lower() == 'fx':
    if location_mode == 'file':
        warn_msg = "File mode: no valid lat/lon provided; skipping model domain bounds check."
        log_message(warn_msg, logfile)
    elif stn_lat is not None and stn_lon is not None:
        bounds = MODEL_BOUNDS.get(model)
        if bounds:
            if not (bounds['lat_min'] <= stn_lat <= bounds['lat_max'] and bounds['lon_min'] <= stn_lon <= bounds['lon_max']):
                err_msg = f"Point ({stn_lat:.2f}, {stn_lon:.2f}) is outside {model} domain boundaries: "
                err_msg += f"lat {bounds['lat_min']:.1f}–{bounds['lat_max']:.1f}°, lon {bounds['lon_min']:.1f}–{bounds['lon_max']:.1f}°"
                log_message(err_msg, logfile)
                raise ValueError(err_msg)


# Helper function to determine data source
def determine_data_source(skew_type, model, station_found):
    """
    Determine which data source to use based on configuration.
    Returns: 'uw' (UW observations), 'eccc_datamart' (RDPS from Datamart), or 'geomet' (HRDPS/GDPS from GeoMet)
    """
    if skew_type.lower() == 'obs':
        return 'uw'
    elif skew_type.lower() == 'fx':
        if model == 'RDPS' and station_found:
            return 'eccc_datamart'
        else:
            return 'geomet'
    else:
        raise ValueError(f"Unknown skew_type: {skew_type}")

# Build a URL to grab the csv output from UW 
def get_uw_ua(date, hour, stn_id):

    # URL format for UW csv observed UA is
    # https://weather.uwyo.edu/wsgi/sounding?datetime=2025-12-23%2012:00:00&id=71964&type=TEXT:CSV
    date_formatted = f'{date[0:4]}-{date[4:6]}-{date[6:8]}'
    url = f'https://weather.uwyo.edu/wsgi/sounding?datetime={date_formatted}%20{hour}:00:00&id={stn_id}&type=TEXT:CSV'
    return url


# Build a URL to get data from ECCC
def get_eccc_ua(date, hour, stn_id):

    # URL format for ECCC csv forecast UA is
    # https://dd.weather.gc.ca/20251223/WXO-DD/vertical_profile/forecast/csv/ProgTephi_12_71964.csv
    url = f'https://dd.weather.gc.ca/{date}/WXO-DD/vertical_profile/forecast/csv/ProgTephi_{hour}_{stn_id.upper()}.csv'
    return url


# ------ reshape_uw
# Reshape the UW data into a usable df
def reshape_uw_df(df_raw):
   
    # Rename columns as required
    df_raw = df_raw.rename(columns={'geopotential height_m':'geopotential height_dm'})
   
    # Convert non-numeric data to NaN in key columns
    key_cols = ['pressure_hPa', 'temperature_C', 'dew point temperature_C', 'wind speed_m/s', 'wind direction_degree']
    for col in key_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    df_raw['wind speed_kmh'] = df_raw['wind speed_m/s'].values.astype(float) * 3.6 * units.km / units.h

    # Remove rows with NaN in key columns
    df_raw = df_raw.dropna(subset=['pressure_hPa', 'temperature_C', 'dew point temperature_C', 'relative humidity_%'])

    # Reset index
    df_all = df_raw.reset_index(drop=True)

    return df_all
    

# ----- Function to reshape the ECCC data into something usable
def reshape_eccc_df(df_raw, fx_hours=[0, 6, 12, 18, 24, 36, 48]):

    df_raw = df_raw.rename(columns={'Variable': 'variable', 'Level': 'pressure', 'Fcst_Hr': 'forecast_hour', 'Value': 'value'})
    df_raw = df_raw.sort_values(['forecast_hour', 'pressure'], ascending=[True, False])
    df_raw = df_raw[df_raw['pressure'].between(100, 1015)]

    dfs = {}
    for fh in sorted(df_raw['forecast_hour'].unique()):
        df_fh = df_raw[df_raw['forecast_hour'] == fh]
        df_pivot = df_fh.pivot(index='pressure', columns='variable', values='value').reset_index()
        df_pivot = df_pivot.sort_values('pressure')

        # Rename columns
        rename_dict = {
            'pressure': 'pressure_hPa',
            'TT': 'temperature_C',
            'WD': 'wind direction_degree',
            'UV': 'wind speed_kt',
            'ES': 'dewpoint_depression_C',
            'GZ': 'geopotential height_dm'
        }
        df_pivot = df_pivot.rename(columns=rename_dict)

        # Calculate dewpoint
        if 'temperature_C' in df_pivot.columns and 'dewpoint_depression_C' in df_pivot.columns:
            df_pivot['dew point temperature_C'] = df_pivot['temperature_C'] - df_pivot['dewpoint_depression_C']
        
        dfs[fh] = df_pivot

    # Combine all forecast hours into one df
    all_dfs = []
    for fh, df_pivot in dfs.items():
        df_pivot['forecast_hour'] = fh
        all_dfs.append(df_pivot)

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Sort: first ascending forecast_hour, then descending pressure
    df_all = df_all.sort_values(['forecast_hour', 'pressure_hPa'], ascending=[True, False])

    # Convert speed from kt to km/h and geo height to dm
    df_all['wind speed_kmh'] = df_all['wind speed_kt'].astype(float) * 1.852
    df_all['geopotential height_dm'] = df_all['geopotential height_dm'].astype(float) * 10

    df_all = df_all[df_all['forecast_hour'].isin(fx_hours)]

    # Remove rows with NaN in key columns
    subset = ['pressure_hPa', 'temperature_C', 'dew point temperature_C']
    if 'relative humidity_%' in df_all.columns:
        subset.append('relative humidity_%')
    df_all = df_all.dropna(subset=subset)

    # Reset index
    df_all = df_all.reset_index(drop=True)

    return df_all


# ------Function to create a name for the output figure
def make_title(site, type, date, hour, fx_hour=None, zoom=False):
    if type == "obs":
        if zoom:
            return f'{site}_{type}_{date}_{hour}UTC_skewT_loweratmos.png'
        return f'{site}_{type}_{date}_{hour}UTC_skewT.png'
    elif type == "fx":
        if zoom:
            return f'{site}_{type}_{date}_{hour}UTCp{fx_hour}_skewT_loweratmos.png'
        return f'{site}_{type}_{date}_{hour}UTCp{fx_hour}_skewT.png'


def resolve_site_prefix(stn_id, stn_iata_code, stn_wmo_id, stn_lat, stn_lon):
    """Choose a site prefix for filenames; fall back to lat/lon when ID is missing.

    - If stn_id is non-empty, use it.
    - Else if IATA code exists, use it.
    - Else build a lat/lon string rounded to 1 decimal with separators removed, e.g., 60.1 -135.2 -> 6011352.
    """
    if stn_id:
        return str(stn_id)
    if stn_iata_code:
        return str(stn_iata_code)
    if stn_lat is not None and stn_lon is not None:
        lat_str = f"{stn_lat:.1f}".replace('.', '').replace('-', '')
        lon_str = f"{stn_lon:.1f}".replace('.', '').replace('-', '')
        return f"{lat_str}{lon_str}"
    return "site"


#------------------------- Plot skew. 
# Some code is from the MetPy Cookbook: https://unidata.github.io/MetPy/latest/examples/skewt_soundings/Skew-T_Soundings.html
# df = a data frame containing pressure (ascending) and corresponding T, Td, WS, WD fields for a single sounding, or for multiple soundings if plot_type = 'all'
# skew_type: <'obs'|'fx'>. String.
# plot_type: <'single'|'all'>. String.  If 'single', the df must contain only one sounding whether it's obs or fx. If 'all', 
#             the script will overplot all soundings in the df and will not overplot sounding calculations (LCL, parcel profile)
# zoom: <True|False>. Boolean. if True, will create a zoomed-in version of the skewt focusing on the lower atmosphere. 
def plot_skewt(df, zoom=False):

    # Remove any data where GZ < 0.0 m
    height_dm = df['geopotential height_dm'].astype(float) 
    mask = height_dm > 0.0
    df = df[mask].reset_index(drop=True)

    # Zoom in if zoom=True
    if zoom:
        df = df[df['pressure_hPa'] >= 500].reset_index(drop=True)
        ylim_bottom = 1000
        ylim_top = 500
        aspect = 250
        rotation = 30
        hodo = False
    else:
        ylim_bottom = 1000
        ylim_top = 100
        aspect = 80.5 #metpy default is 80.5
        rotation = 45 # Default is 30
        hodo = True

    # Populate the required series for plotting
    pres = df['pressure_hPa'].values * units.hPa
    temp = df['temperature_C'].values * units.degC
    dewpoint = df['dew point temperature_C'].values * units.degC
    
    # Generate wind barbs
    wind_speed = df['wind speed_kmh'].values * units.km / units.h
    wind_dir = df['wind direction_degree'].values * units.degrees
    u, v = mpcalc.wind_components(wind_speed, wind_dir)

    # Define the figure and rotation
    fig = plt.Figure(figsize=(9, 9))
    skew = SkewT(fig, rotation=rotation, aspect=aspect)

    # plot t, td, wind
    skew.plot(pres, temp, 'red', label='Temperature')
    skew.plot(pres, dewpoint, 'green', label='Dew Point')

    barb_interval = np.arange(150, 1000, 50) * units('hPa')
    ix = mpcalc.resample_nn_1d(pres, barb_interval)
    skew.plot_barbs(pres[ix], u[ix], v[ix], xloc=1)

    # Calculate and plot LCL and parcel profile
    # Use the lowest (surface) level for parcel calculations
    # pres is sorted descending (highest pressure = surface is first)
    surface_idx = 0
    lcl_pressure, lcl_temperature = mpcalc.lcl(pres[surface_idx], temp[surface_idx], dewpoint[surface_idx])
    skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black', label='LCL')

    profile = mpcalc.parcel_profile(pres, temp[surface_idx], dewpoint[surface_idx]).to('degC')
    skew.plot(pres, profile, 'k', linestyle='dashed', linewidth=2, label='Parcel Profile')

    # Tweak the labels and axes
    skew.ax.set_xlabel('Temperature (°C)')
    skew.ax.set_ylabel('Pressure (hPa)')
    skew.ax.set_ylim(ylim_bottom, ylim_top)
    skew.ax.set_xlim(-55, 30)

    skew.ax.axvline(0, color='c', linestyle='--', linewidth=2, label='0°C Isotherm')

    # Plot adiabats and mixing lines
    skew.plot_dry_adiabats(t0=np.arange(200, 533, 15) * units.K, linewidth=1, alpha=0.3, color='orangered', label ='Dry Adiabats')
    skew.plot_moist_adiabats(linewidth=1, alpha=0.3, color='green', label='Moist Adiabats')
    skew.plot_mixing_lines(pressure=np.arange(1000, 99, -25) * units.hPa, linewidth=1, linestyle='dotted', color='tab:blue', label='Mixing Ratio Lines')

    # Add geopotential height labels on the primary y-axis
    target_pressures = [1000, 900, 850, 800, 750, 700, 500, 300, 200, 100]
    pres_df = df['pressure_hPa'].dropna().values
    height_df = df['geopotential height_dm'].dropna().values
    if len(pres_df) > 1 and len(height_df) > 1:
        for p in target_pressures:
            if p >= pres_df.min() and p <= pres_df.max():
                h = np.interp(p, pres_df[::-1], height_df[::-1])
                h_dm = h / 10  # Convert to decameters
                skew.ax.text(-50, p, f'{h_dm:.0f} dm', fontsize=9, color='gray', ha='left', va='center')

    # Add a hodograph
    if hodo:

        mask = height_df < 12000
        uu = u[mask]
        vv = v[mask]
        height_df_12km = height_df[mask]

        ax_hod = inset_axes(skew.ax, '35%','35%', loc=1)
        h = Hodograph(ax_hod, component_range=80)

        h.add_grid(increment=20, ls='-', lw=1.0, alpha=0.7)
        h.add_grid(increment=10, ls='--', lw=1, alpha=0.3)

        h.ax.set_box_aspect(1)
        h.ax.set_yticklabels([])
        h.ax.set_xticklabels([])
        h.ax.set_xticks([])
        h.ax.set_yticks([])
        h.ax.set_xlabel('')
        h.ax.set_ylabel('')

        plt.xticks(np.arange(0, 0, 1))
        plt.yticks(np.arange(0, 0, 1))

        for i in range(20, 160, 20):
            h.ax.annotate(str(i), (i,0), xytext=(0,2), textcoords='offset pixels', clip_on=True, fontsize=8, weight='normal', alpha=0.5, zorder=0)
        for i in range(20,160,20):
            h.ax.annotate(str(i), (0,i), xytext=(0,2), textcoords='offset pixels', clip_on=True, fontsize=8, weight='normal', alpha=0.5, zorder=0)

        h.plot_colormapped(uu, vv, c=height_df_12km, cmap='viridis', label='0-12km wind')

    # Add legends
    leg_main = skew.ax.legend(loc='upper left')
    
    
    return skew


def plot_and_save_forecasts(df, model_time, model, station_meta, plot_config):
    site_prefix = resolve_site_prefix(plot_config['stn_id'], station_meta['stn_iata_code'], station_meta['stn_wmo_id'], station_meta['stn_lat'], station_meta['stn_lon'])

    for fx_hour in sorted(df['forecast_hour'].unique()):
        df_fh = df[df['forecast_hour'] == fx_hour]
        title = f"{model} Forecast for {station_meta['stn_name']}: Model init {plot_config['date']} {plot_config['hour']}UTC. Valid {model_time + pd.Timedelta(hours=fx_hour)}UTC (+{fx_hour}h)"
        subtitle = f"{station_meta['stn_iata_code']}/{station_meta['stn_wmo_id']}, Lat: {station_meta['stn_lat']:.2f}°, Lon: {station_meta['stn_lon']:.2f}°, Elev: {station_meta['stn_elev']} m"

        skewt = plot_skewt(df_fh, zoom=plot_config['zoom'])

        skewt.ax.set_title(title, fontsize='large', pad=24, loc='center')
        skewt.ax.text(0.5, 1.02, subtitle, fontsize='medium', ha='center', va='bottom', transform=skewt.ax.transAxes)

        current_utc = pd.Timestamp.utcnow()
        add_timestamp(skewt.ax, time=current_utc, y=-0.10, x=0.0, ha='left', time_format='%Y-%m-%d %H:%M UTC', fontsize='medium')

        skewt.ax.text(1.08, 0.5, 'Wind (km/h)', transform=skewt.ax.transAxes, rotation=90, va='center', ha='left', fontsize='medium')

        filename = make_title(site_prefix, plot_config['skew_type'], plot_config['date'], plot_config['hour'], fx_hour, zoom=plot_config['zoom'])
        os.makedirs('figures', exist_ok=True)
        filepath = os.path.join('figures', filename)
        skewt.ax.figure.savefig(filepath)
        log_message(f"Saved figure: {filename}", plot_config['logfile'])


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    # Determine data source based on configuration
    if input_file:
        data_source = 'user_file'
    else:
        data_source = determine_data_source(skew_type, model, station_found)
    log_message(f"Data source: {data_source}, Model: {model}, Skew type: {skew_type}", logfile)

    # === USER-PROVIDED FILE (bypass retrieval) ===
    if data_source == 'user_file':
        df = load_user_profile(input_file, skew_type, logfile)

        if skew_type.lower() == 'fx':
            model_time = pd.to_datetime(f'{date} {hour}', format='%Y%m%d %H')
            station_meta = {
                'stn_name': stn_name,
                'stn_iata_code': stn_iata_code,
                'stn_wmo_id': stn_wmo_id,
                'stn_lat': stn_lat,
                'stn_lon': stn_lon,
                'stn_elev': stn_elev,
                'stn_mod_elev': stn_mod_elev
            }
            plot_config = {
                'skew_type': skew_type,
                'date': date,
                'hour': hour,
                'zoom': zoom,
                'logfile': logfile,
                'stn_id': stn_id
            }
            plot_and_save_forecasts(df, model_time, model, station_meta, plot_config)
        else:
            title = f'User Skew-T valid {date} {hour}UTC'
            subtitle = 'User-provided profile'
            skewt = plot_skewt(df, zoom=zoom)
            skewt.ax.set_title(title, fontsize='large', pad=24, loc='center')
            skewt.ax.text(0.5, 1.02, subtitle, fontsize='medium', ha='center', va='bottom', transform=skewt.ax.transAxes)

            current_utc = pd.Timestamp.utcnow()
            add_timestamp(skewt.ax, time=current_utc, y=-0.10, x=0.0, ha='left', time_format='%Y-%m-%d %H:%M UTC', fontsize='medium')
            skewt.ax.text(1.08, 0.5, 'Wind (km/h)', transform=skewt.ax.transAxes, rotation=90, va='center', ha='left', fontsize='medium')

            site_prefix = resolve_site_prefix(stn_id, stn_iata_code, stn_wmo_id, stn_lat, stn_lon)
            filename = make_title(site_prefix, skew_type, date, hour, zoom=zoom)
            os.makedirs('figures', exist_ok=True)
            filepath = os.path.join('figures', filename)
            skewt.ax.figure.savefig(filepath)
            log_message(f"Saved figure: {filename}", logfile)

    # === OBSERVED SOUNDINGS (UW) ===
    elif data_source == 'uw':
        if stn_wmo_id is None:
            err_msg = "Observed soundings require a valid WMO station ID from station_data.csv"
            log_message(err_msg, logfile)
            raise ValueError(err_msg)
        
        url = get_uw_ua(date, hour, stn_wmo_id)
        try:
            df_raw = pd.read_csv(url, sep=',', header=0)
        except Exception as e:
            err_msg = f"Failed to retrieve observed data from {url}: {e}"
            log_message(err_msg, logfile)
            raise SystemExit("Data retrieval failed")
   
        df = reshape_uw_df(df_raw)
        skew_type_title = 'Observed'
        title = f'{skew_type_title} Skew-T for {stn_name} valid {date} {hour}UTC'
        subtitle = f'{stn_iata_code}/{stn_wmo_id}, Lat: {stn_lat:.2f}°, Lon: {stn_lon:.2f}°, Elev: {stn_elev} m'

        skewt = plot_skewt(df, zoom=zoom)
        
        # Move the main title up and add a subtitle below, both centered
        skewt.ax.set_title(title, fontsize='large', pad=24, loc='center')
        skewt.ax.text(0.5, 1.02, subtitle, fontsize='medium', ha='center', va='bottom', transform=skewt.ax.transAxes)

        current_utc = pd.Timestamp.utcnow()
        add_timestamp(skewt.ax, time=current_utc, y=-0.10, x=0.0, ha='left', time_format='%Y-%m-%d %H:%M UTC', fontsize='medium')

        ## Add label for secondary y-axis (height)
        skewt.ax.text(1.08, 0.5, 'Wind (km/h)', transform=skewt.ax.transAxes, rotation=90, va='center', ha='left', fontsize='medium')

        site_prefix = resolve_site_prefix(stn_id, stn_iata_code, stn_wmo_id, stn_lat, stn_lon)
        filename = make_title(site_prefix, skew_type, date, hour, zoom=zoom)
        os.makedirs('figures', exist_ok=True)
        filepath = os.path.join('figures', filename)
        skewt.ax.figure.savefig(filepath)
        log_message(f"Saved figure: {filename}", logfile)

    # === RDPS FORECAST FROM DATAMART ===
    elif data_source == 'eccc_datamart':
        url = get_eccc_ua(date, hour, stn_iata_code)
        try:
            df_raw = pd.read_csv(url, header=1, skiprows=[2])
        except Exception as e:
            err_msg = f"Failed to retrieve forecast data from {url}: {e}"
            log_message(err_msg, logfile)
            raise SystemExit("Data retrieval failed")
        
        df = reshape_eccc_df(df_raw)

        # Build metadata and config dicts for plotting
        station_meta = {
            'stn_name': stn_name,
            'stn_iata_code': stn_iata_code,
            'stn_wmo_id': stn_wmo_id,
            'stn_lat': stn_lat,
            'stn_lon': stn_lon,
            'stn_elev': stn_elev,
            'stn_mod_elev': stn_mod_elev
        }
        plot_config = {
            'skew_type': skew_type,
            'date': date,
            'hour': hour,
            'zoom': zoom,
            'logfile': logfile,
            'stn_id': stn_id
        }

        model_time = pd.to_datetime(f'{date} {hour}', format='%Y%m%d %H')
        plot_and_save_forecasts(df, model_time, model, station_meta, plot_config)
    
    # === HRDPS/GDPS FORECAST FROM GEOMET ===
    elif data_source == 'geomet':
        log_message(f"Calling get_geomet_ua.py for {model} forecast at lat={stn_lat:.2f}, lon={stn_lon:.2f}", logfile)

        try:
            df = get_geomet_profiles(lat=stn_lat, lon=stn_lon, model=model, time_window_h=48, time_step_h=3)
        except Exception as e:
            err_msg = f"GeoMet retrieval failed for {model} at point ({stn_lat:.2f}, {stn_lon:.2f}): {e}"
            log_message(err_msg, logfile)
            raise SystemExit("Data retrieval failed")

        if df is None or df.empty:
            err_msg = f"GeoMet returned no data for {model} at point ({stn_lat:.2f}, {stn_lon:.2f}) in requested time window (T+0 to T+48h)"
            log_message(err_msg, logfile)
            raise SystemExit("Data retrieval failed")

        # Build metadata and config dicts for plotting
        station_meta = {
            'stn_name': stn_name,
            'stn_iata_code': stn_iata_code,
            'stn_wmo_id': stn_wmo_id,
            'stn_lat': stn_lat,
            'stn_lon': stn_lon,
            'stn_elev': stn_elev,
            'stn_mod_elev': stn_mod_elev
        }
        plot_config = {
            'skew_type': skew_type,
            'date': date,
            'hour': hour,
            'zoom': zoom,
            'logfile': logfile,
            'stn_id': stn_id
        }

        model_time = pd.to_datetime(f'{date} {hour}', format='%Y%m%d %H')
        plot_and_save_forecasts(df, model_time, model, station_meta, plot_config)


