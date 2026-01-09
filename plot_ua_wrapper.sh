#!/bin/bash
# Calling script for plot_ua.py

# User-supplied parameters:
# Start and End dates in the format YYYY-MM-DD, UTC
# Station ID (e.g., CYXY). 

# Requires:
# plot_ua.py: Main data retrieval and plotting script
# get_geomet_ua.py: Called by plot_ua.py for any stations where pre-defined prog SkewT data are not available.
# station_data.csv: A lookup table for station metadata used by plot_ua.py.

# Notes:
# Only retrieves 00z and 12Z times. 
# Does not currently accept lat lon in place of station ID.

# Start and end dates
start_date="2025-12-06"
end_date="2026-01-08"

# Station ID. Limited for now to a 4-letter IATA code as listed in station_data.csv
station_id="CYXY"

# Logfile. Change if desired.
logfile="logs/retrieve_soundings.log"

# Convert to seconds since epoch for iteration
current=$(date -d "$start_date" +%s)
end=$(date -d "$end_date" +%s)

echo "Script: $0"                   > $logfile
echo "Logfile: $logfile"           >> $logfile
echo "==========================================" >> $logfile
echo "Start time: $(date)"          >> $logfile
echo "Retrieving soundings from $start_date to $end_date for station $station_id" >> $logfile
echo "==========================================" >> $logfile

# Loop through each day
while [ $current -le $end ]; do
    # Format date as YYYYMMDD
    date_str=$(date -d "@$current" +%Y%m%d)
    
    # Run for 00Z
    echo "Processing: $date_str 00Z" >> $logfile
    python3 plot_ua.py --stn_id "$station_id" --date "$date_str" --hour 00 --skew_type obs
    
    # Run for 12Z
    echo "Processing: $date_str 12Z" >> $logfile
    python3 plot_ua.py --stn_id "$station_id" --date "$date_str" --hour 12 --skew_type obs
    
    # Move to next day (86400 seconds = 1 day)
    current=$((current + 86400))
done

echo "==========================================" >> $logfile
echo "Script complete at $(date)." >> $logfile
