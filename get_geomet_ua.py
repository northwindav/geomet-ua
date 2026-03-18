# get_geomet_ua.py
# Smith Dec 2025
# Retrieves upper air forecast data from models available on MSC Geomet

# Requires: pandas, numpy, requests, xarray, metpy lxml, netCDF4

# To-do (Thread-safety):
# get_geomet_profiles() currently mutates module-level globals (POINT, TIME_WINDOW_H, TIME_STEP_H, PROFILE_CACHE, caches) 
# and restores them. This is not thread-safe for parallel calls. Refactor to:
#   1. Accept context dict with point/time params instead of mutating globals
#   2. Return cache state with results instead of using module-level cache
#   3. Consider using queue.Queue or concurrent.futures for thread-safe cache access
#   4. Or use process-level isolation if parallelizing across models/points
#
# Docs:
# - GeoMet overview & usage: https://eccc-msc.github.io/open-data/msc-geomet/readme_en/
# - GDPS on GeoMet (GetCapabilities and time/elevation tips): https://eccc-msc.github.io/open-data/msc-data/nwp_gdps/readme_gdps-geomet_en/
# - WMS GetFeatureInfo point sampling & raw data options: https://eccc-msc.github.io/open-data/usage/readme_en/
# - WCS example (image/netcdf): https://pavics-sdi.readthedocs.io/en/latest/notebooks/WCS_example.html
# - HRDPS Open Government entry (variables via GeoMet/WMS): https://open.canada.ca/data/en/dataset/5b401fa0-6c29-57f0-b3d5-749f301d829d

import os, math, re, time, concurrent.futures, threading
import hashlib, pickle
import numpy as np
import pandas as pd
import requests
import xarray as xr
from xml.etree import ElementTree as ET

# MetPy for unit conversions and calculations
import metpy.calc as mpcalc
from metpy.units import units

# -------------------- CONFIG --------------------
POINT = {"lat": 60.1, "lon": -135.0}  # Overridable entry point for lat/lon
OUTDIR = "out"
TIME_STEP_H = 3 # 3-hour timesteps should be sufficient under normal conditions, but 1-hour is available for some models
TIME_WINDOW_H = 48  # Max time to retrieve data.
WMS_URL = "https://geo.weather.gc.ca/geomet"  # GeoMet-Weather WMS

# Performance parameters
# Because we have to pull tiles via WMS, specify how big of a tile to pull for each model type. Smaller is better.
MODEL_BBOX_DEG = {
    "HRDPS": 0.03,   # ~2.5km grid → 3x grid cells for safety
    "RDPS": 0.12,    # ~10km grid → ~1.2x grid cell
    "GDPS": 0.18     # ~15km grid → ~1.2x grid cell
}
# Reduced canvas for point extraction
CANVAS_PX = (50, 50)  # Smaller canvas sufficient for point sampling

# REQUEST VERTICAL SLICES (all pressure levels in one WCS call)
# This reduces API calls from 11/variable to 1/variable per timestep but as of March 2026 doesn't work
USE_VERTICAL_SLICE = False  # Set False to revert to single-level requests 

# USE WMS DIRECTLY
USE_WMS_DIRECT = True  # Set True to skip WCS attempts and use WMS GetFeatureInfo directly. Avoids the inevitable failover from trying WCS as of spring 2026.

# REQUEST CACHING
# Cache requests so that data aren't re-downloaded for repeated plots.
ENABLE_CACHE = True
CACHE_DIR = os.path.join(OUTDIR, ".cache")  # Cache location
CACHE_EXPIRY_HOURS = 1  # Cache expiration time in hours. Tune based on need (forecast data updates ~4x/day)

# CAPABILITIES CACHING
# Cache WMS GetCapabilities responses to avoid re-fetching layer metadata
CACHE_CAPABILITIES = True
CAPABILITIES_CACHE_EXPIRY_HOURS = 24  

# -------------------- GLOBAL SESSION (Connection Pooling) --------------------
# Reuse a shared HTTP session for connection pooling and keep-alive; avoids repeated TLS handshakes
# and reduces latency across the many GeoMet requests this script makes. 
# note: This makes a huuuuuge difference in run-time.
SESSION = requests.Session()

# Threading lock for non-thread-safe scipy operations (e.g., integration)
SCIPY_LOCK = threading.Lock()

# counters for performance monitoring.
WMS_CALLS = 0
CACHE_HITS = 0
CACHE_MISSES = 0

# Variable names (confirmed from GeoMet for all models)
# Note: Wind direction uses WDIR (GDPS), WD (HRDPS, RDPS)
VARIABLE_NAMES = {
    "T": "TT",            # Temperature
    "DEPR": "ES",         # Dewpoint depression (all models use ES)
    "WSPD": "WSPD",       # Wind speed
    "WDIR": ["WDIR", "WD"],  # Wind direction (both variants)
    "GZ": "GZ"            # Geopotential height
}

# Fallback layer names for geopotential height (model heights) if discovery misses them
GZ_LAYER_FALLBACK = {
    "GDPS": "GDPS.ETA_GZ",
    "RDPS": "RDPS.ETA_GZ",
    "HRDPS": "HRDPS.CONTINENTAL_GZ"
}

# -------------------- CACHE for OVERPLOT --------------------
PROFILE_CACHE = {}  # key: (model, valid_time), value: DataFrame profile with derived fields (z_m, etc.)

def ensure_dir(path): os.makedirs(path, exist_ok=True)

def make_cache_key(layer, time_iso, lat, lon, bbox, levels=None):
    """Generate unique cache key from request parameters."""
    # Round coords to avoid floating-point precision issues
    key_parts = [
        str(layer),
        str(time_iso),
        f"{lat:.4f}",
        f"{lon:.4f}",
        f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
    ]
    if levels:
        key_parts.append(",".join(map(str, sorted(levels))))
    
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cached_response(cache_key):
    """Retrieve cached response if available and not expired."""
    global CACHE_HITS, CACHE_MISSES
    
    if not ENABLE_CACHE:
        return None
    
    ensure_dir(CACHE_DIR)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    if not os.path.exists(cache_file):
        CACHE_MISSES += 1
        return None
    
    # Check cache age
    cache_age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
    if cache_age_hours > CACHE_EXPIRY_HOURS:
        CACHE_MISSES += 1
        try:
            os.remove(cache_file)  # Remove expired cache
        except:
            pass
        return None
    
    # Load cached data
    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        CACHE_HITS += 1
        return cached_data
    except Exception:
        CACHE_MISSES += 1
        return None

def save_cached_response(cache_key, data):
    """Save response to cache."""
    if not ENABLE_CACHE:
        return
    
    ensure_dir(CACHE_DIR)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception:
        pass  # Cache write failure shouldn't break the script

def get_cached_capabilities(model):
    """Retrieve cached capabilities XML if available and not expired."""
    if not CACHE_CAPABILITIES:
        return None
    
    ensure_dir(CACHE_DIR)
    cache_file = os.path.join(CACHE_DIR, f".caps_{model}.pkl")
    
    if not os.path.exists(cache_file):
        return None
    
    # Check cache age
    cache_age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
    if cache_age_hours > CAPABILITIES_CACHE_EXPIRY_HOURS:
        try:
            os.remove(cache_file)
        except:
            pass
        return None
    
    # Load cached data
    try:
        with open(cache_file, 'rb') as f:
            caps_xml = pickle.load(f)
        return caps_xml
    except Exception:
        return None

def save_cached_capabilities(model, caps_xml):
    """Save capabilities XML to cache."""
    if not CACHE_CAPABILITIES:
        return
    
    ensure_dir(CACHE_DIR)
    cache_file = os.path.join(CACHE_DIR, f".caps_{model}.pkl")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(caps_xml, f)
    except Exception:
        pass


def wind_dir_speed_from_uv(u_ms, v_ms):
    spd_ms = math.hypot(u_ms, v_ms)
    wdir = (270 - math.degrees(math.atan2(v_ms, u_ms))) % 360  # meteorological FROM direction
    return wdir, spd_ms

def wind_components_threadsafe(speed, direction):
    """Thread-safe wrapper for metpy.calc.wind_components (scipy not thread-safe)."""
    with SCIPY_LOCK:
        return mpcalc.wind_components(speed, direction)

def get_time_dimension(model):
    """Fetch time dimension from any available layer for the model. Caches XML response."""
    ns = {"wms":"http://www.opengis.net/wms"}
    
    # Try cached capabilities first
    caps_xml = get_cached_capabilities(model)
    if caps_xml is None:
        url = f"{WMS_URL}?service=WMS&version=1.3.0&request=GetCapabilities"
        r = SESSION.get(url, timeout=60)
        r.raise_for_status()
        caps_xml = ET.fromstring(r.text)
        save_cached_capabilities(model, caps_xml)
    
    # Find first PRES_ layer for this model and extract time dimension
    for lyr in caps_xml.iterfind(".//wms:Layer/wms:Layer", ns):
        name_el = lyr.find("wms:Name", ns)
        if name_el is None: continue
        name = (name_el.text or "")
        if model.upper() not in name.upper() or "PRES_" not in name: continue
        
        # Found a layer for this model, get its time dimension
        dims = {}
        for tag in ("Dimension","Extent"):
            for el in lyr.findall(f"wms:{tag}", ns):
                k = el.attrib.get("name","").lower()
                vals = (el.text or "").strip()
                if vals:
                    dims[k] = vals.split(",")
        return caps_xml, dims
    
    raise RuntimeError(f"No pressure-level layers found for {model}")

def find_layers(caps_xml, model, var_names_dict):
    """Scan capabilities, find layer names containing model and variables.
    var_names_dict: dict like {"T": "TT", "DEPR": "ES", "WDIR": ["WDIR", "WD"]}
    Returns: dict like {"T": [...layer names...], "DEPR": [...], ...}
    """
    ns = {"wms":"http://www.opengis.net/wms"}
    found = {k: [] for k in var_names_dict.keys()}
    
    for lyr in caps_xml.iterfind(".//wms:Layer/wms:Layer", ns):
        name_el = lyr.find("wms:Name", ns)
        if name_el is None: continue
        name = (name_el.text or "")
        if model.upper() not in name.upper(): continue
        
        # Skip surface-only, GEML, WEonG, and contour layers
        if any(x in name.lower() for x in ['weon', '2m', 'surface', 'contour', 'geml']):
            continue
        # Skip height AGL layers (_40m, _80m, _120m) - we need pressure levels
        if re.search(r'_\d+m$', name):
            continue
        # Accept isobaric layers: either _XXXmb suffix or \.XXX suffix
        has_mb_suffix = re.search(r'_\d+mb$', name)
        has_dot_suffix = re.search(r'\.\d+$', name)
        if not (has_mb_suffix or has_dot_suffix):
            continue
            
        # Check each variable token
        for var_key, var_spec in var_names_dict.items():
            # var_spec can be a string like "TT" or list like ["WDIR", "WD"]
            variants = var_spec if isinstance(var_spec, list) else [var_spec]
            
            for variant in variants:
                # Match variant in layer name (case insensitive)
                if variant.upper() in name.upper() or f"_{variant.upper()}" in name.upper() or f".{variant.upper()}_" in name.upper():
                    found[var_key].append(name)
                    break  # Found match for this variable, stop checking variants
    
    return found

def layer_dims(caps_xml, layer_name):
    """Return dict of dimensions: {time: [...], elevation: [...] if present}."""
    ns = {"wms":"http://www.opengis.net/wms"}
    for lyr in caps_xml.iterfind(".//wms:Layer/wms:Layer", ns):
        nm = lyr.find("wms:Name", ns)
        if nm is None or nm.text != layer_name: continue
        dims = {}
        for tag in ("Dimension","Extent"):
            for el in lyr.findall(f"wms:{tag}", ns):
                k = el.attrib.get("name","").lower()
                vals = (el.text or "").strip()
                if vals:
                    dims[k] = vals.split(",")
        return dims
    return {}


def parse_time_values(raw_list):
    """Expand WMS time extents (intervals or discrete) to list of UTC pandas Timestamps."""
    out = []
    for val in raw_list:
        if not val: continue
        val = val.strip()
        if 'T:' in val:
            val = val.replace('T:', 'T00:')  # fix malformed hour-less tokens sometimes seen
        if '/' in val:
            parts = val.split('/')
            if len(parts) == 3:
                start, end, step = parts
                try:
                    step_td = pd.to_timedelta(step)
                    rng = pd.date_range(start=start, end=end, freq=step_td, tz="UTC")
                    out.extend(rng)
                    continue
                except Exception:
                    pass
        try:
            ts = pd.Timestamp(val)
            # timestamps from WMS may already have 'Z' timezone - convert/localize appropriately
            if ts.tz is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            out.append(ts)
        except Exception:
            continue
    return out

# Retrieve raster coverage and cache requests
def wcs_getcoverage(layer, time_iso, lat, lon, bbox, size_px, extra_params=None):
    global WMS_CALLS
    
    # If USE_WMS_DIRECT is enabled, skip WCS entirely and go straight to WMS
    if USE_WMS_DIRECT:
        return wms_getfeatureinfo_fallback(layer, time_iso, lat, lon, bbox, size_px, extra_params)
    
    # Check cache first
    cache_key = make_cache_key(layer, time_iso, lat, lon, bbox)
    cached = get_cached_response(cache_key)
    if cached is not None:
        print(f"[WCS cache hit] layer={layer} time={time_iso} lat={lat:.3f} lon={lon:.3f}")
        return cached
    
    WMS_CALLS += 1
    miny, minx, maxy, maxx = bbox
    width, height = size_px
    
    try:
        # WCS uses array-style subset parameters
        url = f"{WMS_URL}?service=WCS&version=2.0.1&request=GetCoverage&coverageId={layer}&format=image/tiff"
        url += f"&subset=Lat({miny},{maxy})&subset=Long({minx},{maxx})&subset=time(\"{time_iso}\")"
        url += f"&size=x({width}),y({height})"
        
        r = SESSION.get(url, timeout=30)
    except Exception as e:
        print(f"[WCS error] layer={layer} time={time_iso} lat={lat:.3f} lon={lon:.3f} err={e}")
        return np.nan

    if not r.ok:
        print(f"[WCS HTTP {r.status_code}] layer={layer} time={time_iso} bbox={bbox}")
        return np.nan
    
    # Parse GeoTIFF response
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(r.content))
        arr = np.array(img)
        
        # Extract pixel at (lon, lat) -> column i, row j
        i = int(round((lon - minx) / (maxx - minx) * width))
        j = int(round((lat - miny) / (maxy - miny) * height))
        
        # Clamp to valid range
        i = np.clip(i, 0, arr.shape[1] - 1)
        j = np.clip(j, 0, arr.shape[0] - 1)
        
        val = float(arr[j, i])
        # GeoTIFF may have no-data values; GDPS/RDPS typically use -9999 or similar
        if val < -1000:
            val = np.nan
        
        # Cache response (even if nan - avoids re-fetching bad data)
        save_cached_response(cache_key, val)
        return val
    except Exception as e:
        print(f"[WCS parse fallback] layer={layer} time={time_iso} lat={lat:.3f} lon={lon:.3f} err={e}")
        # Fallback: try text/plain if GeoTIFF fails
        result = wms_getfeatureinfo_fallback(layer, time_iso, lat, lon, bbox, size_px, extra_params)
        # Cache fallback result too
        save_cached_response(cache_key, result)
        return result

# This is the ideal method, but as of spring 2026 does not work on geomet. 
# Current settings skip this call entirely, but it can be used and will fail
# over to WMS if unsuccessful.
def wcs_get_vertical_profile(layer_template, time_iso, lat, lon, bbox, pressure_levels):

    global WMS_CALLS
    
    # Check cache first
    cache_key = make_cache_key(layer_template, time_iso, lat, lon, bbox, pressure_levels)
    cached = get_cached_response(cache_key)
    if cached is not None:
        return cached
    
    WMS_CALLS += 1
    
    miny, minx, maxy, maxx = bbox
    
    # Build layer name from template (assumes no pressure-specific suffix)
    # Extract base layer name (remove pressure suffix if present)
    layer_base = re.sub(r'[._]\d+(mb)?$', '', layer_template)
    
    # Construct WCS request for elevation range
    min_p = min(pressure_levels)
    max_p = max(pressure_levels)
    
    base_url = f"{WMS_URL}?service=WCS&version=2.0.1&request=GetCoverage"
    base_url += f"&coverageId={layer_base}"
    base_url += f"&format=application/x-netcdf"  # NetCDF better for multi-dimensional data
    base_url += f"&subset=Lat({miny},{maxy})&subset=Long({minx},{maxx})"
    base_url += f"&subset=time(\"{time_iso}\")"

    # Try pressure-based subset first; fall back to elevation if needed
    subset_dims = ["pressure", "isobaric", "elevation"]
    response = None
    for subset_dim in subset_dims:
        url = f"{base_url}&subset={subset_dim}({min_p},{max_p})"
        try:
            r = SESSION.get(url, timeout=60)
        except Exception as e:
            print(f"[WCS vertical error] layer={layer_base} subset={subset_dim} time={time_iso} err={e}")
            continue
        if not r.ok:
            print(f"[WCS vertical HTTP {r.status_code}] layer={layer_base} subset={subset_dim} time={time_iso}")
            continue
        response = r
        break

    if response is None:
        save_cached_response(cache_key, None)
        return None

    try:
        # Parse NetCDF response with xarray
        import io
        # Try scipy engine first (more commonly available), then h5netcdf
        ds = None
        for engine in ['scipy', 'h5netcdf', None]:
            try:
                ds = xr.open_dataset(io.BytesIO(response.content), engine=engine)
                break
            except (ValueError, OSError) as engine_err:
                # Engine not available or data format incompatible, try next
                if engine is None:
                    # Last attempt failed
                    raise engine_err
                continue
        
        if ds is None:
            return None
        
        # Extract point values for all pressure levels
        # NetCDF structure varies; look for pressure/elevation dimension
        result = {}
        
        # Find the data variable (skip coordinate vars)
        data_vars = [v for v in ds.data_vars if 'lat' not in str(v).lower() and 'lon' not in str(v).lower()]
        if not data_vars:
            return None
            
        var = ds[data_vars[0]]
        
        # Find pressure/elevation coordinate
        elev_coord = None
        for coord in ['isobaric', 'pressure', 'elevation', 'level']:
            if coord in var.dims or coord in ds.coords:
                elev_coord = coord
                break
        
        if not elev_coord:
            return None
        
        # Extract nearest point to (lat, lon)
        point_data = var.sel(lat=lat, lon=lon, method='nearest')
        
        # Get values for each pressure level
        if elev_coord in point_data.dims:
            for p in pressure_levels:
                try:
                    val = float(point_data.sel({elev_coord: p}, method='nearest').values)
                    if np.isfinite(val) and val > -1000:  # Filter no-data values
                        result[int(p)] = val
                except Exception:
                    continue
        
        ds.close()
        
        # Cache response (even if empty - avoid re-attempting)
        save_cached_response(cache_key, result if result else None)
        
        return result if result else None
        
    except Exception as e:
        # NetCDF parsing failed; cache None to avoid re-attempting
        # Only log unexpected errors (suppress common engine/format issues that are handled by fallback)
        err_str = str(e).lower()
        suppress_patterns = ['engine', 'unrecognized', 'not a valid netcdf']
        if not any(pattern in err_str for pattern in suppress_patterns):
            print(f"[WCS vertical parse fail] layer={layer_base} time={time_iso} err={e}")
        save_cached_response(cache_key, None)
        return None

# Mis-named, because this is the primary function used to retrieve data from Geomet as the
#  single WCS call does not currently work with Geomet.
def wms_getfeatureinfo_fallback(layer, time_iso, lat, lon, bbox, size_px, extra_params=None):

    miny, minx, maxy, maxx = bbox
    width, height = size_px
    i = int(round((lon - minx) / (maxx - minx) * width))
    j = int(round((lat - miny) / (maxy - miny) * height))
    params = {
        "service": "WMS", "version": "1.3.0", "request": "GetFeatureInfo",
        "layers": layer, "query_layers": layer,
        "crs": "EPSG:4326",
        "bbox": f"{miny},{minx},{maxy},{maxx}",
        "width": width, "height": height,
        "info_format": "text/plain",
        "time": time_iso,
        "i": i, "j": j
    }
    if extra_params:
        params.update(extra_params)
    
    try:
        r = SESSION.get(WMS_URL, params=params, timeout=30)
    except Exception:
        return np.nan
    
    if not r.ok: 
        return np.nan
    
    for line in r.text.splitlines():
        # Look for actual value fields (not metadata like x, y, class, red, green, blue)
        line_lower = line.lower()
        if ("value" in line_lower or "band" in line_lower) and "=" in line:
            try:
                val_str = line.split("=")[-1].strip().strip("'\"")  # Remove quotes
                return float(val_str)
            except Exception:
                pass
    return np.nan


# Clean up layer accessor names. 
def make_level_accessor(caps_xml, layer_name, model=None):

    dims = layer_dims(caps_xml, layer_name)
    if any(k in dims for k in ("elevation", "pressure")):
        dim_key = "elevation" if "elevation" in dims else "pressure"
        return lambda p: (layer_name, {dim_key: int(p)})

    # Check for dot-suffix pattern (e.g., GDPS.PRES_TT.1000 -> GDPS.PRES_TT.500)
    m = re.search(r"\.(\d+)$", layer_name)
    if m:
        pre = layer_name[:m.start(0)]  # everything before the last dot
        
        # GDPS upper levels (100-400 hPa) require .3h suffix
        if model == "GDPS":
            return lambda p: (f"{pre}.{int(p)}.3h" if 100 <= p <= 400 else f"{pre}.{int(p)}", None)
        else:
            return lambda p: (f"{pre}.{int(p)}", None)
    
    # Check for underscore-suffix pattern (e.g., VAR_1000mb -> VAR_500mb)
    m = re.search(r"_(\d+)(mb)?$", layer_name)
    if m:
        pre = layer_name[:m.start(0)]
        return lambda p: (f"{pre}_{int(p)}mb", None)

    # Default: append _XXXmb
    return lambda p: (f"{layer_name}_{int(p)}mb", None)

# --- Met calculations for height ---
def vapor_pressure_from_td(td_c):
    a, b = 17.625, 243.04
    return 6.1094 * np.exp(a*td_c/(b+td_c))

def mixing_ratio_from_e(p_hpa, e_hpa):
    eps = 0.622
    return eps * e_hpa / (p_hpa - e_hpa)

def virtual_temperature(t_k, r):
    return t_k * (1 + 0.61*r)

def hypsometric_thickness(p1_hpa, p2_hpa, Tv_mean_k):
    Rd = 287.05
    g0 = 9.80665
    return (Rd * Tv_mean_k / g0) * np.log(p1_hpa / p2_hpa)

def build_height(profile):
    prof = profile.copy()
    prof["T_K"] = prof["T_C"] + 273.15
    prof["Td_K"] = prof["Td_C"] + 273.15
    prof["e_hPa"] = vapor_pressure_from_td(prof["Td_C"])
    prof["r_kgkg"] = mixing_ratio_from_e(prof["pressure_hpa"], prof["e_hPa"])
    prof["Tv_K"]  = virtual_temperature(prof["T_K"], prof["r_kgkg"])
    prof = prof.sort_values("pressure_hpa", ascending=False).reset_index(drop=True)

    # If model-provided heights are present and finite, honor them; otherwise compute hypsometric
    if "z_m" in prof.columns and prof["z_m"].notna().all():
        return prof

    z = [0.0]
    for i in range(1, len(prof)):
        p1 = prof.loc[i-1, "pressure_hpa"]
        p2 = prof.loc[i,   "pressure_hpa"]
        Tv_mean = 0.5*(prof.loc[i-1,"Tv_K"] + prof.loc[i,"Tv_K"])
        dz = hypsometric_thickness(p1, p2, Tv_mean)
        z.append(z[-1] + float(dz))
    prof["z_m"] = z
    return prof

# Conducts the actual calls
def run_model(model, time_window_h=None, time_step_h=None):
    """Run model data retrieval using layer discovery."""
    tw = int(time_window_h) if time_window_h is not None else TIME_WINDOW_H
    ts = int(time_step_h) if time_step_h is not None else TIME_STEP_H
    model_timer_start = time.perf_counter()
    
    # Fetch capabilities once and extract time dimension
    caps, dims = get_time_dimension(model)
    times = parse_time_values(dims.get("time", []))
    if not times:
        raise RuntimeError(f"[{model}] No time dimension found in capabilities")

    # Discover layer names for required variables
    cand = find_layers(caps, model, VARIABLE_NAMES)
    
    lyr_T    = cand["T"][0] if cand["T"] else None
    lyr_DEPR = cand["DEPR"][0] if cand["DEPR"] else None
    lyr_WSPD = cand["WSPD"][0] if cand["WSPD"] else None
    lyr_WDIR = cand["WDIR"][0] if cand["WDIR"] else None
    
    # Model surface elevation (terrain height) - not pressure-level dependent
    lyr_SURFACE_ELEV = GZ_LAYER_FALLBACK.get(model)

    # Verify we found required layers
    if not all([lyr_T, lyr_DEPR, lyr_WSPD, lyr_WDIR]):
        raise RuntimeError(f"[{model}] Missing required layers: T={bool(lyr_T)}, DEPR={bool(lyr_DEPR)}, "
                          f"WSPD={bool(lyr_WSPD)}, WDIR={bool(lyr_WDIR)}")

    # choose T0→T? in configurable steps (subset if fewer times available)
    t0 = times[0]
    desired = [t0 + pd.Timedelta(hours=h) for h in range(0, tw+1, ts)]
    chosen = [t for t in desired if t in times]
    if not chosen:  # fallback: stride through whatever is available
        chosen = times[::ts] or times

    # Pressure levels to fetch
    levels = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]

    # Create layer accessors (pass model for GDPS-specific handling)
    acc_T    = make_level_accessor(caps, lyr_T, model=model)
    acc_DEPR = make_level_accessor(caps, lyr_DEPR, model=model)
    acc_WS   = make_level_accessor(caps, lyr_WSPD, model=model)
    acc_WD   = make_level_accessor(caps, lyr_WDIR, model=model)

    # Spatial setup - use model-specific BBOX
    bbox_deg = MODEL_BBOX_DEG.get(model, 0.1)  # Default to 0.1 if model not configured
    half = bbox_deg / 2.0
    bbox = (POINT["lat"]-half, POINT["lon"]-half, POINT["lat"]+half, POINT["lon"]+half)
    w, h = CANVAS_PX

    # Thread pool for concurrent requests (optimized from testing: 24 workers)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=24)

    for t_iso in chosen:
        time_str = t_iso.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Query model surface elevation once per timestep (not pressure-dependent)
        surface_elev_m = np.nan
        if lyr_SURFACE_ELEV:
            surface_elev_m = wcs_getcoverage(lyr_SURFACE_ELEV, time_str, POINT["lat"], POINT["lon"], bbox, (w,h), None)
        
        # OPTIMIZATION: Try to fetch entire vertical profile in single WCS call
        # As of spring 2026 fails, so suggest keeping USE_VERTICAL_SLICE = FALSE
        # until when and if this feature functions on geomet.
        rows = []
        if USE_VERTICAL_SLICE:
            # Attempt multi-level retrieval for each variable
            T_profile = wcs_get_vertical_profile(lyr_T, time_str, POINT["lat"], POINT["lon"], bbox, levels)
            DEPR_profile = wcs_get_vertical_profile(lyr_DEPR, time_str, POINT["lat"], POINT["lon"], bbox, levels)
            WSPD_profile = wcs_get_vertical_profile(lyr_WSPD, time_str, POINT["lat"], POINT["lon"], bbox, levels)
            WDIR_profile = wcs_get_vertical_profile(lyr_WDIR, time_str, POINT["lat"], POINT["lon"], bbox, levels)
            
            # If core profiles retrieved successfully, combine them
            if all(p is not None for p in [T_profile, DEPR_profile, WSPD_profile, WDIR_profile]):
                # Combine profiles by pressure level
                for p in levels:
                    if p not in T_profile or p not in DEPR_profile:
                        continue
                    
                    T_c = T_profile[p]
                    depr = DEPR_profile[p]
                    Td_c = T_c - depr
                    
                    # Sanity check
                    if Td_c > T_c + 0.1 or not np.isfinite(Td_c):
                        continue
                    
                    wspd_val = WSPD_profile.get(p)
                    wdir_val = WDIR_profile.get(p)
                    if wspd_val is None or wdir_val is None:
                        continue
                    
                    wspd_si = float(wspd_val)
                    wdir_deg = float(wdir_val) % 360.0
                    u_ms, v_ms = wind_components_threadsafe(wspd_si * units.meter/units.second,
                                                       wdir_deg * units.degree)
                    wdir, wspd = wind_dir_speed_from_uv(float(u_ms.magnitude), float(v_ms.magnitude))
                    wspd_kmh = wspd * 3.6
                    
                    row = {"pressure_hpa": int(p), "T_C": T_c, "Td_C": Td_c,
                           "wind_dir_deg": wdir, "wind_spd_ms": wspd, "wind_spd_kmh": wspd_kmh,
                           "model_surface_elev_m": surface_elev_m}
                    rows.append(row)
        
        # Script will fail into using WMS every time so we use WMS to request each pressure level instead.
        if not rows:
            def fetch_row(p):
                def q(accessor):
                    lid, extra = accessor(p)
                    return wcs_getcoverage(lid, time_str, POINT["lat"], POINT["lon"], bbox, (w,h), extra)

                T_c = q(acc_T)
                if not np.isfinite(T_c):
                    return None

                # Dewpoint from depression (available for all models)
                depr = q(acc_DEPR)
                Td_c = T_c - depr if np.isfinite(depr) else np.nan

                if not np.isfinite(Td_c):
                    return None

                # Sanity check: Td should not exceed T
                if Td_c > T_c + 0.1:
                    return None  # Skip bad data instead of raising

                # Wind from speed/direction
                wspd_val = q(acc_WS)
                wdir_val = q(acc_WD)
                if not all(np.isfinite(x) for x in [T_c, Td_c, wspd_val, wdir_val]):
                    return None
                
                wspd_si = float(wspd_val)
                wdir_deg = float(wdir_val) % 360.0
                u_ms, v_ms = wind_components_threadsafe(wspd_si * units.meter/units.second,
                                                   wdir_deg * units.degree)
                wdir, wspd = wind_dir_speed_from_uv(float(u_ms.magnitude), float(v_ms.magnitude))
                wspd_kmh = wspd * 3.6

                row = {"pressure_hpa": int(p), "T_C": T_c, "Td_C": Td_c,
                       "wind_dir_deg": wdir, "wind_spd_ms": wspd, "wind_spd_kmh": wspd_kmh,
                       "model_surface_elev_m": surface_elev_m}
                return row

            futures = [executor.submit(fetch_row, p) for p in levels]

            for f in futures:
                res = f.result()
                if res is not None:
                    rows.append(res)

        if not rows:
            print(f"[{model}] {t_iso} — no data rows assembled (all NaN or missing).")
            continue

        df = pd.DataFrame(rows).sort_values("pressure_hpa", ascending=False)
        if df.empty:
            print(f"[{model}] {t_iso} — no data rows assembled.")
            continue

        # Derived vertical coordinates & venting/inversion summary
        prof = build_height(df)

        # Cache
        PROFILE_CACHE[(model, t_iso)] = prof.copy()

    executor.shutdown(wait=True)
    
    model_elapsed = time.perf_counter() - model_timer_start
    # Return all profiles collected for this model
    return {"profiles": dict(PROFILE_CACHE), "elapsed": model_elapsed}


# Master function, imported by plot_ua.py. This function calls all related retrieval and caching functions.
def get_geomet_profiles(lat, lon, model, time_window_h=None, time_step_h=None):
    #   Returns a pd.DataFrame containing:
    #    Columns: pressure_hPa, temperature_C, dew point temperature_C,
    #    wind direction_degree, wind speed_kmh, geopotential height_dm,
    #    forecast_hour.

    global POINT, PROFILE_CACHE, WMS_CALLS, CACHE_HITS, CACHE_MISSES

    # Save existing point and override for this call
    orig_point = POINT.copy()

    POINT = {"lat": float(lat), "lon": float(lon)}
    tw = int(time_window_h) if time_window_h is not None else TIME_WINDOW_H
    ts = int(time_step_h) if time_step_h is not None else TIME_STEP_H

    # Reset per-call caches to avoid stale profiles
    PROFILE_CACHE = {}
    WMS_CALLS = 0
    CACHE_HITS = 0
    CACHE_MISSES = 0

    try:
        result = run_model(model, time_window_h=tw, time_step_h=ts)
        profiles = result.get("profiles", {})
        if not profiles:
            return pd.DataFrame()

        # Determine base time (first available) for forecast hour offsets
        times = [t for (_, t) in profiles.keys()]
        base_time = min(times)

        records = []
        for (_, t_iso), prof in profiles.items():
            fh = int((t_iso - base_time).total_seconds() // 3600)
            df = prof.copy()
            df = df.rename(columns={
                "pressure_hpa": "pressure_hPa",
                "T_C": "temperature_C",
                "Td_C": "dew point temperature_C",
                "wind_dir_deg": "wind direction_degree",
                "wind_spd_kmh": "wind speed_kmh"
            })
            # geopotential height in decameters for consistency with plotting code
            df["geopotential height_dm"] = df.get("z_m", np.nan) / 10.0
            # model surface elevation in decameters
            df["model_surface_elev_dm"] = df.get("model_surface_elev_m", np.nan) / 10.0
            df["forecast_hour"] = fh
            records.append(df[[
                "pressure_hPa",
                "temperature_C",
                "dew point temperature_C",
                "wind direction_degree",
                "wind speed_kmh",
                "geopotential height_dm",
                "model_surface_elev_dm",
                "forecast_hour"
            ]].dropna(subset=["pressure_hPa", "temperature_C", "dew point temperature_C"]))

        if not records:
            return pd.DataFrame()
        return pd.concat(records, ignore_index=True)

    finally:
        # Restore globals
        POINT = orig_point

