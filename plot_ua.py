#!/usr/bin/env python3
"""CLI entrypoint for sounding retrieval and plotting.

This script delegates workflow orchestration to modules in ua_core.
"""

import argparse
from datetime import datetime, timezone

from ua_core.config import DEFAULT_CONFIG_PATH, load_config
from ua_core.pipeline import RequestOptions, close_log, init_log, run_request


def parse_args():
    today_utc = datetime.now(timezone.utc).strftime("%Y%m%d")

    parser = argparse.ArgumentParser(description="Plot upper air soundings from BUFR/UW/GeoMet")
    parser.add_argument("--stn_id", type=str, default="cyxy", help="Station ID (WMO or ICAO code)")
    parser.add_argument("--date", type=str, default=today_utc, help="Date in YYYYMMDD format")
    parser.add_argument(
        "--hour",
        type=str,
        default="00",
        choices=["00", "06", "12", "18"],
        help="Observation or model init hour (UTC)",
    )
    parser.add_argument("--skew_type", type=str, default="obs", choices=["obs", "fx"], help="Type of sounding: obs or fx")
    parser.add_argument("--zoom", action="store_true", help="Zoom to lower atmosphere")
    parser.add_argument("--logfile", type=str, default=None, help="Logfile path for output messages")
    parser.add_argument(
        "--location_mode",
        type=str,
        default="station",
        choices=["station", "latlon", "file"],
        help="Location mode: station ID, latlon, or file input",
    )
    parser.add_argument("--lat", type=float, default=None, help="Latitude in decimal degrees")
    parser.add_argument("--lon", type=float, default=None, help="Longitude in decimal degrees")
    parser.add_argument(
        "--model",
        type=str,
        default="HRDPS",
        choices=["HRDPS", "RDPS", "GDPS"],
        help="Model for forecast requests",
    )
    parser.add_argument("--input_file", type=str, default=None, help="CSV profile input to bypass retrieval")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config JSON")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate arguments and pipeline setup only; skip data retrieval and plotting",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    logfile = args.logfile or cfg["paths"]["default_logfile"]
    init_log(logfile)

    opts = RequestOptions(
        stn_id=args.stn_id,
        date=args.date,
        hour=args.hour,
        skew_type=args.skew_type.lower(),
        zoom=args.zoom,
        logfile=logfile,
        location_mode=args.location_mode,
        lat=args.lat,
        lon=args.lon,
        model=args.model.upper(),
        input_file=args.input_file,
        validate_only=args.validate_only,
    )

    try:
        run_request(opts, cfg)
    finally:
        close_log(logfile)


if __name__ == "__main__":
    main()
