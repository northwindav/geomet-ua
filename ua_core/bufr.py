import re
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from pybufrkit.decoder import Decoder

from decode_bufr import (
    convert_profile_units,
    extract_station_metadata,
    filter_profile_by_elevation,
    profile_dataframe_from_long,
)


def _iter_subsets(template_data):
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

    for subset_idx, (descriptors, values) in enumerate(zip(descriptors_all, values_all)):
        yield subset_idx, descriptors, values


def _extract_template_data(bufr_message):
    template_data = bufr_message.template_data

    if hasattr(template_data, "value") and template_data.value is not None:
        return template_data.value

    if hasattr(template_data, "get_value"):
        value = template_data.get_value()
        if value is not None:
            return value

    section4 = getattr(bufr_message, "section4", None)
    if section4 is not None and hasattr(section4, "template_data"):
        section_td = section4.template_data
        if hasattr(section_td, "value") and section_td.value is not None:
            return section_td.value
        if section_td is not None:
            return section_td

    return template_data


def decode_bufr_bytes_to_dataframe(content: bytes) -> pd.DataFrame:
    decoder = Decoder()
    bufr_message = decoder.process(content)

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


def _folder_hour_for_bulletin(valid_hour: str, bulletin: str) -> str:
    hour_int = int(valid_hour)
    if bulletin in {"IUS", "IUJ"}:
        return f"{(hour_int + 1) % 24:02d}"
    return f"{hour_int:02d}"


def _render_template(template: str, mapping: Dict[str, str]) -> str:
    rendered = template
    for token, value in mapping.items():
        rendered = rendered.replace("{" + token + "}", value)
    return rendered


def _extract_file_href_candidates(index_text: str) -> List[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', index_text, flags=re.IGNORECASE)
    if hrefs:
        return hrefs

    # Fallback for simple/plain indexes where filenames may appear as bare text.
    return re.findall(r"[A-Z]{3,4}_[A-Z0-9]{3,5}_[0-9]{6}___[0-9]+", index_text)


def _discover_bufr_file_url(
    session: requests.Session,
    folder_url: str,
    bulletin: str,
    station_upper: str,
    timeout: int,
) -> Optional[Tuple[str, str]]:
    response = session.get(folder_url, timeout=timeout)
    response.raise_for_status()

    hrefs = _extract_file_href_candidates(response.text)
    # Datamart bulletin names often include a variant token (e.g. IUKB01, IUWB01).
    pattern = re.compile(rf"^{bulletin}[A-Z0-9]*_{station_upper}_[0-9]{{6}}___[0-9]+$")
    matches = []
    for href in hrefs:
        filename = href.split("/")[-1].strip()
        if pattern.match(filename):
            matches.append(filename)

    if not matches:
        return None

    selected = sorted(set(matches))[-1]
    return urljoin(folder_url + "/", selected), selected


def _select_bulletin_order(date: str, hour: str, now_utc: pd.Timestamp) -> List[str]:
    valid_dt = pd.Timestamp(f"{date} {hour}:00", tz="UTC")
    iuk_ready = now_utc >= (valid_dt + timedelta(minutes=15))
    ius_ready = now_utc >= (valid_dt + timedelta(minutes=75))

    if ius_ready:
        return ["IUS", "IUK"]
    if iuk_ready:
        return ["IUK"]
    return []


def try_fetch_bufr_observed_profile(
    date: str,
    hour: str,
    station_id: str,
    cfg: Dict,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[str]]:
    if not cfg.get("enabled", False):
        return None, None, None

    url_template = cfg.get("url_template", "")
    if not url_template:
        return None, None, None

    timeout = int(cfg.get("timeout_seconds", 20))
    station_upper = str(station_id).upper()
    now_utc = pd.Timestamp.now(tz="UTC")
    bulletins = _select_bulletin_order(date, hour, now_utc)

    if not bulletins:
        return None, None, None

    session = requests.Session()

    ddhhmm = f"{date[-2:]}{hour}00"

    for bulletin in bulletins:
        folder_hour = _folder_hour_for_bulletin(hour, bulletin)
        rendered = _render_template(
            url_template,
            {
                "date": date,
                "station": station_upper,
                "hour": folder_hour,
                "ddHHMM": ddhhmm,
                "bulletin": bulletin,
                "IUK|IUW|IUJ|IUS": bulletin,
                "random digits": "",
            },
        )

        folder_url = rendered.rsplit("/", 1)[0]
        try:
            discovered = _discover_bufr_file_url(
                session=session,
                folder_url=folder_url,
                bulletin=bulletin,
                station_upper=station_upper,
                timeout=timeout,
            )
            if not discovered:
                continue

            url, filename = discovered
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            long_df = decode_bufr_bytes_to_dataframe(response.content)

            station_elev_m, stn_lat, stn_lon = extract_station_metadata(long_df)
            profile_df = profile_dataframe_from_long(long_df)
            profile_df = filter_profile_by_elevation(profile_df, station_elev_m)
            profile_df = convert_profile_units(profile_df)

            # Harmonize names with the plotting pipeline.
            profile_df = profile_df.rename(
                columns={
                    "pressure": "pressure_hPa",
                    "Geopotential_height": "geopotential height_dm",
                    "Temperature": "temperature_C",
                    "Dewpoint_Temperature": "dew point temperature_C",
                    "Wind_direction": "wind direction_degree",
                    "Wind_speed": "wind speed_kmh",
                }
            )

            metadata = {
                "stn_elev": int(station_elev_m) if station_elev_m is not None else 0,
                "stn_lat": float(stn_lat) if stn_lat is not None else None,
                "stn_lon": float(stn_lon) if stn_lon is not None else None,
                "bufr_bulletin": bulletin,
                "bufr_filename": filename,
                "bufr_url": url,
            }
            return profile_df, metadata, filename
        except Exception:
            continue

    return None, None, None
