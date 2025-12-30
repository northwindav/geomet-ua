# Download any instance of the rdps surface geopotential height from the datamart and
# match model elevations with lat/lon points in a list of stations, named station_list.csv
# The station_list.csv file should have columns: station_id, lat, lon, elev (elevation in meters)
# The output will be a new csv file named station_data_with_model_elev.csv with an added column model_elev

# Required grib file is named (as of Dec 2025): <YYYYMMDDTHH>Z_MSC_RDPS_GeopotentialHeight_Sfc_RLatLon0.09_PT000H.grib2
import pandas as pd
import pygrib
import numpy as np
import os

# Load station data
station_df = pd.read_csv('station_list.csv')

# Open the grib2 file (replace with actual filename if different)
grib_file = [f for f in os.listdir('.') if f.endswith('.grib2')][0]
grbs = pygrib.open(grib_file)

# List all variables in the grib file
print('Variables in the grib2 file:')
for grb in grbs:
    print(f"{grb.name} | shortName: {grb.shortName} | typeOfLevel: {grb.typeOfLevel}")
grbs.rewind()

# Find the field for surface geopotential height or elevation
for grb in grbs:
    if 'geopotential height' in grb.name.lower() or 'surface' in grb.name.lower() or 'orography' in grb.name.lower():
        elev_grb = grb
        break
else:
    raise ValueError("No suitable elevation field found in the grib2 file.")

# Get data and lats/lons
data, lats, lons = elev_grb.data()

# Interpolate model elevation for each station
def get_model_elev(lat, lon):
    # Find the nearest grid point
    dist = np.sqrt((lats - lat)**2 + (lons - lon)**2)
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return float(data[idx])

station_df['model_elev'] = station_df.apply(lambda row: get_model_elev(row['lat'], row['lon']), axis=1)

# Save to a new file to avoid overwriting
station_df.to_csv('station_data_with_model_elev.csv', index=False)