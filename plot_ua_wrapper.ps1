# User-facing wrapper for plot_ua.py (PowerShell version)

param(
    [Alias('k')]
    [string]$SkewType = "obs",
    
    [Alias('d')]
    [string]$Date = "",
    
    [Alias('H')]
    [string]$Hour = "00",
    
    [Alias('m')]
    [string]$Model = "hrdps",
    
    [Alias('s')]
    [string]$StationId = "fake",
    
    [string]$Lat = "61.1",
    [string]$Lon = "-132.2",
    
    [Alias('f')]
    [string]$InputFile = "",
    
    [Alias('l')]
    [string]$Logfile = "logs/retrieve_soundings.log",
    
    [switch]$Help
)

function Show-Usage {
    Write-Host @"
Usage: plot_ua_wrapper.ps1 [options]

Options (user-specified):
  -k, -SkewType     fx|obs (default: obs)
  -d, -Date         Date in YYYY-MM-DD (UTC)
  -H, -Hour         Hour in UTC (00, 06, 12, 18). Default: 00
  -m, -Model        HRDPS | RDPS | GDPS | all (default: hrdps)
  -s, -StationId    Station ID (3-5 alphanumeric IATA. E.g. CYEG for Edmonton)
      -Lat          Latitude (used when station-id omitted)
      -Lon          Longitude (used when station-id omitted)
  -f, -InputFile    Path to CSV with required columns (bypasses data retrieval)
  -l, -Logfile      Logfile path (default: logs/retrieve_soundings.log)
  -h, -Help         Show this help

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
"@
}

if ($Help) {
    Show-Usage
    exit 0
}

# Set default date to current UTC date if not provided
if ([string]::IsNullOrEmpty($Date)) {
    $Date = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd")
}

# Validate skew type
$SkewType = $SkewType.ToLower()
if ($SkewType -ne "fx" -and $SkewType -ne "obs") {
    Write-Error "skew-type must be fx or obs"
    Show-Usage
    exit 1
}

# Validate hour
if ($Hour -notmatch "^(00|06|12|18)$") {
    Write-Error "hour must be one of 00, 06, 12, 18"
    Show-Usage
    exit 2
}

# Validate and process model selection
$Model = $Model.ToUpper()
if ($Model -eq "ALL") {
    $Models = @("HRDPS", "RDPS", "GDPS")
} elseif ($Model -match "^(HRDPS|RDPS|GDPS)$") {
    $Models = @($Model)
} else {
    Write-Error "model must be HRDPS, RDPS, GDPS, or all"
    Show-Usage
    exit 3
}

# Determine location mode
$LocationMode = ""
if (-not [string]::IsNullOrEmpty($InputFile)) {
    if (-not (Test-Path $InputFile)) {
        Write-Error "Input file not found: $InputFile"
        Show-Usage
        exit 7
    }
    $LocationMode = "file"
} elseif (-not [string]::IsNullOrEmpty($StationId)) {
    if ($StationId -notmatch "^[A-Za-z0-9]{3,5}$") {
        Write-Error "station-id must be 3-5 alphanumeric characters"
        Show-Usage
        exit 6
    }
    $LocationMode = "station"
} elseif (-not [string]::IsNullOrEmpty($Lat) -and -not [string]::IsNullOrEmpty($Lon)) {
    $LocationMode = "latlon"
} else {
    Write-Error "Provide either a station-id, both lat and lon, or an input file"
    Show-Usage
    exit 4
}

# Validate obs requires station
if ($SkewType -eq "obs" -and $LocationMode -eq "latlon") {
    Write-Error "Observed soundings currently require station-id. Forecasts may be requested for any point in the model domain"
    Show-Usage
    exit 5
}

# Prep values for plot_ua.py
$DateCompact = $Date -replace "-", ""
$LogDir = Split-Path -Parent $Logfile
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Initialize logfile (overwrite if exists)
$LogContent = @"
Script: $PSCommandPath
Logfile: $Logfile
==========================================
Start time: $((Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss"))
Selections: skew_type=$SkewType date=$Date hour=$Hour models=$($Models -join ' ')
"@

if ($LocationMode -eq "station") {
    $LogContent += "`nStation ID: $StationId"
} elseif ($LocationMode -eq "latlon") {
    $LogContent += "`nLat/Lon: $Lat, $Lon"
} else {
    $LogContent += "`nInput file: $InputFile"
}

$LogContent += "`n=========================================="

# Write initial log content
Set-Content -Path $Logfile -Value $LogContent

# Execute for each requested model
foreach ($ModelItem in $Models) {
    Add-Content -Path $Logfile -Value "Processing model: $ModelItem"
    
    $PythonArgs = @(
        "plot_ua.py",
        "--date", $DateCompact,
        "--hour", $Hour,
        "--skew_type", $SkewType,
        "--logfile", $Logfile,
        "--location_mode", $LocationMode,
        "--model", $ModelItem
    )
    
    if ($LocationMode -eq "station") {
        $PythonArgs += "--stn_id", $StationId
        # Always pass lat/lon if available for fallback when station not in CSV
        if (-not [string]::IsNullOrEmpty($Lat) -and -not [string]::IsNullOrEmpty($Lon)) {
            $PythonArgs += "--lat", $Lat
            $PythonArgs += "--lon", $Lon
        }
    } elseif ($LocationMode -eq "latlon") {
        $PythonArgs += "--stn_id", ""
        $PythonArgs += "--lat", $Lat
        $PythonArgs += "--lon", $Lon
    } else {
        # File mode
        $PythonArgs += "--stn_id", "custom"
        $PythonArgs += "--input_file", $InputFile
    }
    
    & python $PythonArgs
}

Add-Content -Path $Logfile -Value "=========================================="
Add-Content -Path $Logfile -Value "Script complete at $((Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss"))."
