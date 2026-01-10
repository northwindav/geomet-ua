#!/bin/bash
# User-facing wrapper for plot_ua.py

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: plot_ua_wrapper.sh [options]

Options (user-specified):
  -k, --skew-type   fx|obs (default: obs)
  -d, --date        Date in YYYY-MM-DD (UTC)
  -H, --hour        Hour in UTC (00, 06, 12, 18). Default: 00
  -m, --model       HRDPS | RDPS | GDPS | all (default: hrdps)
  -s, --station-id  Station ID (3-5 alphanumeric IATA. E.g. CYEG for Edmonton)
      --lat         Latitude (used when station-id omitted)
      --lon         Longitude (used when station-id omitted)
  -f, --input-file  Path to CSV with required columns (bypasses data retrieval)
  -l, --logfile     Logfile path (default: logs/retrieve_soundings.log)
  -h, --help        Show this help

Notes:
- Date/hour validity is enforced in plot_ua.py.
- Lat/lon mode requires plot_ua.py support; station ID remains required for obs today.

Exit Codes:
  0 = Success
  1 = Invalid skew-type
  2 = Invalid hour
  3 = Invalid model
  4 = Missing location (station-id or lat/lon)
  5 = Obs mode requires station-id
  6 = Invalid station-id format
    7 = Input file not found
EOF
}

# Defaults
SKEW_TYPE="obs"
DATE_UTC=$(date -u +%Y-%m-%d)
HOUR_UTC="00"
MODEL_SELECTION="hrdps"
STATION_ID=""
LAT=""
LON=""
LOGFILE="logs/retrieve_soundings.log"
INPUT_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -k|--skew-type)
            SKEW_TYPE="$2"; shift 2 ;;
        -d|--date)
            DATE_UTC="$2"; shift 2 ;;
        -H|--hour)
            HOUR_UTC="$2"; shift 2 ;;
        -m|--model)
            MODEL_SELECTION="$2"; shift 2 ;;
        -s|--station-id)
            STATION_ID="$2"; shift 2 ;;
        --lat)
            LAT="$2"; shift 2 ;;
        --lon)
            LON="$2"; shift 2 ;;
        -f|--input-file)
            INPUT_FILE="$2"; shift 2 ;;
        -l|--logfile)
            LOGFILE="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage; exit 1 ;;
    esac
done

# Basic validations with unique error codes
SKEW_TYPE=$(echo "$SKEW_TYPE" | tr '[:upper:]' '[:lower:]')
if [[ "$SKEW_TYPE" != "fx" && "$SKEW_TYPE" != "obs" ]]; then
    echo "skew-type must be fx or obs" >&2
    usage
    exit 1
fi

if [[ ! "$HOUR_UTC" =~ ^(00|06|12|18)$ ]]; then
    echo "hour must be one of 00, 06, 12, 18" >&2
    usage
    exit 2
fi

MODEL_SELECTION=$(echo "$MODEL_SELECTION" | tr '[:lower:]' '[:upper:]')
if [[ "$MODEL_SELECTION" == "ALL" ]]; then
    MODELS=("HRDPS" "RDPS" "GDPS")
elif [[ "$MODEL_SELECTION" =~ ^(HRDPS|RDPS|GDPS)$ ]]; then
    MODELS=("$MODEL_SELECTION")
else
    echo "model must be HRDPS, RDPS, GDPS, or all" >&2
    usage
    exit 3
fi

if [[ -n "$INPUT_FILE" ]]; then
    if [[ ! -f "$INPUT_FILE" ]]; then
        echo "Input file not found: $INPUT_FILE" >&2
        usage
        exit 7
    fi
    LOCATION_MODE="file"
elif [[ -n "$STATION_ID" ]]; then
    if [[ ! "$STATION_ID" =~ ^[A-Za-z0-9]{3,5}$ ]]; then
        echo "station-id must be 3-5 alphanumeric characters" >&2
        usage
        exit 6
    fi
    LOCATION_MODE="station"
elif [[ -n "$LAT" && -n "$LON" ]]; then
    LOCATION_MODE="latlon"
else
    echo "Provide either a station-id, both lat and lon, or an input file" >&2
    usage
    exit 4
fi

if [[ "$SKEW_TYPE" == "obs" && "$LOCATION_MODE" == "latlon" ]]; then
    echo "Observed soundings currently require station-id. Forecasts may be requested for any point in the model domain" >&2
    usage
    exit 5
fi

# Prep values for plot_ua.py
DATE_COMPACT=${DATE_UTC//-/}
LOGDIR=$(dirname "$LOGFILE")
mkdir -p "$LOGDIR"

# Initialize logfile at wrapper start (overwrite if exists), then all subsequent writes append.
# This ensures a fresh log for each wrapper invocation, but all models for this session append to it.
{
    echo "Script: $0"
    echo "Logfile: $LOGFILE"
    echo "=========================================="
    echo "Start time: $(date -u)"
    echo "Selections: skew_type=$SKEW_TYPE date=$DATE_UTC hour=$HOUR_UTC models=${MODELS[*]}"
    if [[ "$LOCATION_MODE" == "station" ]]; then
        echo "Station ID: $STATION_ID"
    elif [[ "$LOCATION_MODE" == "latlon" ]]; then
        echo "Lat/Lon: $LAT, $LON"
    else
        echo "Input file: $INPUT_FILE"
    fi
    echo "=========================================="
} > "$LOGFILE"

# Execute for each requested model.
for MODEL in "${MODELS[@]}"; do
    echo "Processing model: $MODEL" >> "$LOGFILE"

    if [[ "$LOCATION_MODE" == "station" ]]; then
        # Build base command
        CMD="python3 plot_ua.py --stn_id \"$STATION_ID\" --date \"$DATE_COMPACT\" --hour \"$HOUR_UTC\" --skew_type \"$SKEW_TYPE\" --logfile \"$LOGFILE\" --location_mode \"$LOCATION_MODE\" --model \"$MODEL\""
        # Always pass lat/lon if available for fallback when station not in CSV
        if [[ -n "$LAT" && -n "$LON" ]]; then
            CMD="$CMD --lat \"$LAT\" --lon \"$LON\""
        fi
        eval $CMD
    elif [[ "$LOCATION_MODE" == "latlon" ]]; then
        # Lat/lon mode: pass lat/lon for point lookup; stn_id still passed for potential filename use
        python3 plot_ua.py \
            --stn_id "" \
            --date "$DATE_COMPACT" \
            --hour "$HOUR_UTC" \
            --skew_type "$SKEW_TYPE" \
            --logfile "$LOGFILE" \
            --location_mode "$LOCATION_MODE" \
            --lat "$LAT" \
            --lon "$LON" \
            --model "$MODEL"
    else
        # File mode: bypass retrieval, just plot provided data
        python3 plot_ua.py \
            --stn_id "custom" \
            --date "$DATE_COMPACT" \
            --hour "$HOUR_UTC" \
            --skew_type "$SKEW_TYPE" \
            --logfile "$LOGFILE" \
            --location_mode "file" \
            --model "$MODEL" \
            --input_file "$INPUT_FILE"
    fi
done

echo "==========================================" >> "$LOGFILE"
echo "Script complete at $(date -u)." >> "$LOGFILE"
