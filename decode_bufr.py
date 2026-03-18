# decode_BUFR.py
# Smith March 2026
# Coded with assistance from Copilot. Reviewed and edited by author.

# Purpose: Given a BUFR file containing vertical sounding information, decode
# and extract specified fields. 

# External references;
# WMO list of bulletins: https://wis.wmo.int/operational-info/VolumeC1/VolC1.txt
# WMO Manual on the Global telecommunications system. See 'Attachment II-C' (page 80 of 2023 edition) for info pertinent to identifying and decoding BUFR data
# https://library.wmo.int/viewer/35800/download?file=386_2023-edition_en.pdf&type=pdf&navigator=1

# BUFR files for Canadian sites are available on MSC Datamart at:
# https://dd.weather.gc.ca/<YYYYMMDD>/WXO-DD/bulletins/alphanumeric/<YYYYMMDD>/IU/
# where full profiles are available at 00 and 12Z
# Specific BUFR files are:
# IUJxxx: Upper wind profile (not other data)
# IUKxxx: Sounding up to 100 hPa
# IUSxxx: Full sounding
# IUWxxx: Upper wind to 100hPa
# In practice the IUK and IUW are available about 15 minutes after the valid times of 00Z and 12Z, 
#   while the full soundings IUJ and IUS are available about 75 minutes after valid times. 

# Requires pybufrkit.


from typing import Iterable, Tuple

import pandas as pd
from pybufrkit.decoder import Decoder

# Field numbers may be found in any BUFR technical manual, for instance
# https://www.ecmwf.int/sites/default/files/elibrary/2008/80926-bufr-reference-manual_0.pdf
# BUFR Table B shows all possible data codes, while table 3.24 shows the majority, but not all, of the data contained in a CMC upper air observational BUFR file.
TARGET_FIELDS = {
    7004: "pressure",
    10009: "Geopotential_height",
    12101: "Temperature",
    12103: "Dewpoint_Temperature",
    11001: "Wind_direction",
    11002: "Wind_speed",
}

STATION_METADATA_FIELDS = {
    7030: "station_elev_m",
    5001: "Latitude",
    6001: "Longitude",
}


def _iter_subsets(template_data) -> Iterable[Tuple[int, list, list]]:
    descriptors_all = template_data.decoded_descriptors_all_subsets
    values_all = template_data.decoded_values_all_subsets

    if not descriptors_all or not values_all:
        return

    if not isinstance(descriptors_all[0], (list, tuple)):
        descriptors_all = [descriptors_all]
    if not isinstance(values_all[0], (list, tuple)):
        values_all = [values_all]

    if len(descriptors_all) == 1 and len(values_all) > 1:
        descriptors_all = descriptors_all * len(values_all)

    for subset_idx, (descriptors, values) in enumerate(
        zip(descriptors_all, values_all)
    ):
        yield subset_idx, descriptors, values


def _extract_template_data(bufr_message):
    template_data = bufr_message.template_data

    if hasattr(template_data, "value"):
        if template_data.value is not None:
            return template_data.value

    if hasattr(template_data, "get_value"):
        value = template_data.get_value()
        if value is not None:
            return value

    section4 = getattr(bufr_message, "section4", None)
    if section4 is not None:
        if hasattr(section4, "template_data"):
            section_td = section4.template_data
            if hasattr(section_td, "value") and section_td.value is not None:
                return section_td.value
            if section_td is not None:
                return section_td
        if hasattr(section4, "get_parameter"):
            param = section4.get_parameter("template_data")
            if hasattr(param, "value") and param.value is not None:
                return param.value
            if param is not None:
                return param

    return template_data


# Primary function: Decode BUFR into a Pandas df
def decode_bufr_to_dataframe(path: str) -> pd.DataFrame:
    decoder = Decoder()
    with open(path, "rb") as ins:
        bufr_message = decoder.process(ins.read(), file_path=path)

    template_data = _extract_template_data(bufr_message)
    rows = []
    for subset_idx, descriptors, values in _iter_subsets(template_data):
        for value_idx, (descriptor, value) in enumerate(zip(descriptors, values)):
            if value is None:
                continue

            descriptor_id = getattr(descriptor, "id", None)
            if descriptor_id is None:
                descriptor_id = getattr(descriptor, "id_", None)

            rows.append(
                {
                    "subset": subset_idx,
                    "idx": value_idx,
                    "descriptor_id": descriptor_id,
                    "descriptor_name": getattr(descriptor, "name", None),
                    "unit": getattr(descriptor, "unit", None),
                    "value": value,
                }
            )

    return pd.DataFrame(rows)

# Reorder and parse through the df to pull only values that we have specified
#    and add a header.
def profile_dataframe_from_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered_ids = set(TARGET_FIELDS.keys())

    for subset in sorted(df["subset"].unique()):
        subset_df = df[df["subset"] == subset].sort_values("idx")
        current = None

        for _, row in subset_df.iterrows():
            descriptor_id = row.get("descriptor_id")
            if descriptor_id not in ordered_ids:
                continue

            if descriptor_id == 7004:
                if current and any(value is not None for value in current.values()):
                    rows.append(current)
                current = {name: None for name in TARGET_FIELDS.values()}
                current[TARGET_FIELDS[descriptor_id]] = row.get("value")
                continue

            if current is None:
                current = {name: None for name in TARGET_FIELDS.values()}

            col_name = TARGET_FIELDS[descriptor_id]
            if current.get(col_name) is None:
                current[col_name] = row.get("value")

        if current and any(value is not None for value in current.values()):
            rows.append(current)

    return pd.DataFrame(rows, columns=list(TARGET_FIELDS.values()))

# Convert units to hPa, dm, C, km/h
def convert_profile_units(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    converted["pressure"] = converted["pressure"] / 10.0
    converted["Geopotential_height"] = converted["Geopotential_height"] / 10.0
    converted["Temperature"] = converted["Temperature"] - 273.15
    converted["Dewpoint_Temperature"] = (
        converted["Dewpoint_Temperature"] - 273.15
    )
    converted["Wind_speed"] = converted["Wind_speed"] * 3.6
    return converted


# Extract metadata fields
def extract_station_metadata(long_df: pd.DataFrame) -> tuple:
    """Extract station elevation, latitude, longitude from long dataframe."""
    station_elev_m = None
    stn_lat = None
    stn_lon = None

    for _, row in long_df.iterrows():
        descriptor_id = row.get("descriptor_id")

        if descriptor_id == 7030 and station_elev_m is None:
            station_elev_m = row.get("value")
        elif descriptor_id == 5001 and stn_lat is None:
            stn_lat = row.get("value")
        elif descriptor_id == 6001 and stn_lon is None:
            stn_lon = row.get("value")

        if station_elev_m is not None and stn_lat is not None and stn_lon is not None:
            break

    return station_elev_m, stn_lat, stn_lon


# Some BUFR files contain data extrapolate to below station elevation. Remove these rows. 
def filter_profile_by_elevation(df: pd.DataFrame, station_elev_m: float) -> pd.DataFrame:
    if station_elev_m is None:
        return df
    return df[df["Geopotential_height"] >= station_elev_m].reset_index(drop=True)


if __name__ == "__main__":
    
    # Extract BUFR data into a Pandas df
    long_df = decode_bufr_to_dataframe("IUKB01_CYXY_150000___29384")
    
    # Populate metadata fields. Additional fields are available if desired. 
    station_elev_m, stn_lat, stn_lon = extract_station_metadata(long_df)
    
    # Thin and re-order the df to pull only fields specified in TARGET_FIELDS,
    
    profile_df = profile_dataframe_from_long(long_df)
    
    # Drop any data below the station elevation
    profile_df = filter_profile_by_elevation(profile_df, station_elev_m)
    
    # Convert units to eexpected units.
    converted_df = convert_profile_units(profile_df)
    
    # These lines can be removed during normal use: 
    # Print station metadata and first 58 rows of converted profile dataframe for verification
    print(f"Station elevation: {station_elev_m} m")
    print(f"Station latitude: {stn_lat}°")
    print(f"Station longitude: {stn_lon}°")
    print()
    print(converted_df.head(58))

