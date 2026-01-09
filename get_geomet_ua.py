# get_geomet_ua.py
# Smith Dec 2025
# Mucking around to see how easily we can retrieve model forecast data for fx skewTs

# Requires: pandas, numpy, requests, xarray, metpy lxml, netCDF4

# Overplot examples:
# Overplot all times on a single figure
# OVERPLOT = {"enable": True, "by_time": True, "by_model": False}
# Overplot multiple timesteps for a single model
# OVERPLOT = {"enable": True, "by_time": False, "by_model": True}

# T0→T48 every 3h (or until last available time). Includes venting/inversion summary per time.
# Optional overplot: overlay models at same valid time, or overlay timesteps for same model.
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
POINT = {"lat": 60.1, "lon": -135.0} 
OUTDIR = "out"
MODELS = ["HRDPS", "RDPS", "GDPS"]  # test all three models
TIME_STEP_H = 3
TIME_WINDOW_H = 48
WMS_URL = "https://geo.weather.gc.ca/geomet"  # GeoMet-Weather WMS

# PERFORMANCE OPTIMIZATIONS
# Model-specific BBOX (degrees) based on grid spacing (HRDPS: ~2.5km, RDPS: ~10km, GDPS: ~15km)
MODEL_BBOX_DEG = {
    "HRDPS": 0.03,   # ~2.5km grid → 3x grid cells for safety
    "RDPS": 0.12,    # ~10km grid → ~1.2x grid cell
    "GDPS": 0.18     # ~15km grid → ~1.2x grid cell
}
# Reduced canvas for point extraction
CANVAS_PX = (50, 50)  # Smaller canvas sufficient for point sampling

# REQUEST VERTICAL SLICES (all pressure levels in one WCS call)
# This reduces API calls from 11/variable to 1/variable per timestep
USE_VERTICAL_SLICE = True  # Set False to revert to single-level requests

# REQUEST CACHING
# Cache API responses to disk to avoid redundant requests across runs
# Especially useful during development/testing or when re-running with different plot options
ENABLE_CACHE = True
CACHE_DIR = os.path.join(OUTDIR, ".cache")  # Cache location
CACHE_EXPIRY_HOURS = 1  # Cache expiration time in hours. Tune based on need (forecast data updates ~4x/day)

# CAPABILITIES CACHING
# Cache WMS GetCapabilities responses to avoid re-fetching layer metadata
CACHE_CAPABILITIES = True
CAPABILITIES_CACHE_EXPIRY_HOURS = 24  # Capabilities rarely change; cache 24h

# Colors per model for overplot
MODEL_COLORS = {"HRDPS": "tab:red", "GDPS": "tab:blue", "RDPS": "tab:green"}

# -------------------- GLOBAL SESSION (Connection Pooling) --------------------
# Reuse HTTP session for all requests to enable connection pooling
SESSION = requests.Session()  # Automatically handles keep-alive and connection reuse

# Threading lock for non-thread-safe scipy operations (e.g., integration)
SCIPY_LOCK = threading.Lock()

# counters
WMS_CALLS = 0
CACHE_HITS = 0
CACHE_MISSES = 0

# Hardcoded layer name templates for each model (confirmed from GeoMet Capabilities)
# All follow pattern: {MODEL_PREFIX}PRES_{VARIABLE}.{PRESSURE}
MODEL_LAYER_TEMPLATES = {
    "GDPS": "GDPS.PRES_{var}.{pressure}",
    "HRDPS": "HRDPS.CONTINENTAL.PRES_{var}.{pressure}",
    "RDPS": "RDPS.PRES_{var}.{pressure}"
}

# Variable names (confirmed from GeoMet for all models)
# Note: Wind direction uses WDIR (GDPS), WD (HRDPS, RDPS)
VARIABLE_NAMES = {
    "T": "TT",           # Temperature
    "DEPR": "ES",        # Dewpoint depression (all models use ES)
    "WSPD": "WSPD",      # Wind speed
    "WDIR": ["WDIR", "WD"]  # Wind direction (both variants)
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
    except Exception as e:
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

def magnus_dewpoint_c(t_c, rh_pct):
    """Dewpoint from T (°C) and RH (%) via Magnus (over water)."""
    a, b = 17.625, 243.04
    rh = np.clip(rh_pct, 0.1, 100.0)
    gamma = np.log(rh/100.0) + (a*t_c)/(b + t_c)
    return (b*gamma)/(a - gamma)

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

def wcs_getcoverage(layer, time_iso, lat, lon, bbox, size_px, extra_params=None):
    """
    WCS GetCoverage -> retrieve raster coverage, extract value at lat/lon.
    Much more reliable than WMS GetFeatureInfo for point sampling.
    EPSG:4326 axis order: (lat, lon).
    Includes request caching to avoid redundant API calls.
    """
    global WMS_CALLS
    
    # Check cache first
    cache_key = make_cache_key(layer, time_iso, lat, lon, bbox)
    cached = get_cached_response(cache_key)
    if cached is not None:
        return cached
    
    WMS_CALLS += 1
    miny, minx, maxy, maxx = bbox
    width, height = size_px
    
    params = {
        "service": "WCS", "version": "2.0.1", "request": "GetCoverage",
        "coverageId": layer,
        "format": "image/tiff",
        "subsettingCrs": "http://www.opengis.net/gml/srs/epsg.xml#4326",
        "subset": [
            f"Lat({miny},{maxy})",
            f"Long({minx},{maxx})"
        ],
        "size": f"x({width}),y({height})"
    }
    if time_iso:
        params["subset"].append(f"time(\"{time_iso}\")")
    if extra_params:
        params.update(extra_params)
    
    try:
        # WCS uses array-style subset parameters
        url = f"{WMS_URL}?service=WCS&version=2.0.1&request=GetCoverage&coverageId={layer}&format=image/tiff"
        url += f"&subset=Lat({miny},{maxy})&subset=Long({minx},{maxx})&subset=time(\"{time_iso}\")"
        url += f"&size=x({width}),y({height})"
        
        r = SESSION.get(url, timeout=30)
    except Exception as e:
        return np.nan
    
    if not r.ok:
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
        # Fallback: try text/plain if GeoTIFF fails
        result = wms_getfeatureinfo_fallback(layer, time_iso, lat, lon, bbox, size_px, extra_params)
        # Cache fallback result too
        save_cached_response(cache_key, result)
        return result

def wcs_get_vertical_profile(layer_template, time_iso, lat, lon, bbox, size_px, pressure_levels, model):
    """
    OPTIMIZED: Request entire vertical profile (all pressure levels) in a single WCS call.
    Returns dict: {pressure_hpa: value} for all requested levels.
    Uses NetCDF format to retrieve multi-level data efficiently.
    Includes request caching to avoid redundant API calls.
    """
    global WMS_CALLS
    
    # Check cache first
    cache_key = make_cache_key(layer_template, time_iso, lat, lon, bbox, pressure_levels)
    cached = get_cached_response(cache_key)
    if cached is not None:
        return cached
    
    WMS_CALLS += 1
    
    miny, minx, maxy, maxx = bbox
    width, height = size_px
    
    # Build layer name from template (assumes no pressure-specific suffix)
    # Extract base layer name (remove pressure suffix if present)
    layer_base = re.sub(r'[._]\d+(mb)?$', '', layer_template)
    
    # Construct WCS request for elevation range
    min_p = min(pressure_levels)
    max_p = max(pressure_levels)
    
    url = f"{WMS_URL}?service=WCS&version=2.0.1&request=GetCoverage"
    url += f"&coverageId={layer_base}"
    url += f"&format=application/x-netcdf"  # NetCDF better for multi-dimensional data
    url += f"&subset=Lat({miny},{maxy})&subset=Long({minx},{maxx})"
    url += f"&subset=time(\"{time_iso}\")"
    
    # Try elevation subset (some layers use 'elevation', others 'pressure')
    # Format: subset=elevation(min,max) in hPa
    url += f"&subset=elevation({min_p},{max_p})"
    
    try:
        r = SESSION.get(url, timeout=60)
        if not r.ok:
            # Fallback: elevation dimension might not support ranges
            return None
            
        # Parse NetCDF response with xarray
        import io
        ds = xr.open_dataset(io.BytesIO(r.content), engine='h5netcdf')
        
        # Extract point values for all pressure levels
        # NetCDF structure varies; look for pressure/elevation dimension
        result = {}
        
        # Find the data variable (skip coordinate vars)
        data_vars = [v for v in ds.data_vars if 'lat' not in v.lower() and 'lon' not in v.lower()]
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
        save_cached_response(cache_key, None)
        return None

def wms_getfeatureinfo_fallback(layer, time_iso, lat, lon, bbox, size_px, extra_params=None):
    """
    WMS GetFeatureInfo fallback -> numeric value at lat/lon (raw value).
    EPSG:4326 axis order in WMS 1.3.0 is (lat, lon).
    """
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
    except Exception as e:
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

def pick_first(cand_dict, keys):
    for k in keys:
        lst = cand_dict.get(k, [])
        if lst: return lst[0]
    return None

def make_level_accessor(caps_xml, layer_name):
    """Return callable that builds layer name for a given pressure level."""
    dims = layer_dims(caps_xml, layer_name)
    if any(k in dims for k in ("elevation", "pressure")):
        dim_key = "elevation" if "elevation" in dims else "pressure"
        return lambda p: (layer_name, {dim_key: int(p)})

    # Check for dot-suffix pattern (e.g., GDPS.PRES_TT.1000 -> GDPS.PRES_TT.500)
    m = re.search(r"\.(\d+)$", layer_name)
    if m:
        pre = layer_name[:m.start(0)]  # everything before the last dot
        return lambda p: (f"{pre}.{int(p)}", None)
    
    # Check for underscore-suffix pattern (e.g., VAR_1000mb -> VAR_500mb)
    m = re.search(r"_(\d+)(mb)?$", layer_name)
    if m:
        pre = layer_name[:m.start(0)]
        return lambda p: (f"{pre}_{int(p)}mb", None)

    # Default: append _XXXmb
    return lambda p: (f"{layer_name}_{int(p)}mb", None)

# --- Met calculations for height, inversion, venting ---
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
    z = [0.0]
    for i in range(1, len(prof)):
        p1 = prof.loc[i-1, "pressure_hpa"]
        p2 = prof.loc[i,   "pressure_hpa"]
        Tv_mean = 0.5*(prof.loc[i-1,"Tv_K"] + prof.loc[i,"Tv_K"])
        dz = hypsometric_thickness(p1, p2, Tv_mean)
        z.append(z[-1] + float(dz))
    prof["z_m"] = z
    return prof

def potential_temperature(t_k, p_hpa):
    p0 = 1000.0
    Rd_cp = 0.2854
    return t_k * (p0/p_hpa)**Rd_cp

def detect_inversion(prof, max_depth_m=2000.0):
    p = prof.sort_values("z_m")
    for i in range(len(p)-1):
        dz = p.loc[i+1,"z_m"] - p.loc[i,"z_m"]
        if dz <= 0: continue
        dT = p.loc[i+1,"T_C"] - p.loc[i,"T_C"]
        if dT > 0 and p.loc[i,"z_m"] <= max_depth_m:
            j = i+1
            while j < len(p)-1 and (p.loc[j+1,"T_C"] - p.loc[j,"T_C"]) > 0:
                j += 1
            base_z = p.loc[i,"z_m"]; top_z = p.loc[j,"z_m"]
            dT_total = p.loc[j,"T_C"] - p.loc[i,"T_C"]
            return base_z, top_z, dT_total
    return None

def current_mixing_height(prof):
    p = prof.sort_values("z_m")
    theta_sfc = potential_temperature(p.loc[0,"T_K"], p.loc[0,"pressure_hpa"])
    theta_env = potential_temperature(p["T_K"].values, p["pressure_hpa"].values)
    idxs = np.where(theta_env - theta_sfc <= 1.0)[0]
    if len(idxs) == 0: return 0.0
    return float(p.loc[idxs[-1],"z_m"])

def warming_needed_to_break(prof, base_z, top_z):
    p = prof.sort_values("z_m")
    theta_sfc = potential_temperature(p.loc[0,"T_K"], p.loc[0,"pressure_hpa"])
    j = (p["z_m"] - top_z).abs().idxmin()
    theta_top = potential_temperature(p.loc[j,"T_K"], p.loc[j,"pressure_hpa"])
    return float(theta_top - theta_sfc)

def crossover_temperature_from_td(td_c, target_rh_pct):
    a, b = 17.625, 243.04
    e = vapor_pressure_from_td(td_c)
    es_needed = e / (target_rh_pct/100.0)
    ln_term = np.log(es_needed/6.1094)
    T = (b * ln_term) / (a - ln_term)
    return float(T)

def ventilation_index(mix_h_m, mean_wspd_ms):
    return float(mix_h_m * mean_wspd_ms)

# -------------------- MAIN WORKFLOW (GeoMet WMS) --------------------
def run_model(model):
    """Run model data retrieval using layer discovery."""
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

    # Verify we found all required layers
    if not all([lyr_T, lyr_DEPR, lyr_WSPD, lyr_WDIR]):
        raise RuntimeError(f"[{model}] Missing required layers: T={bool(lyr_T)}, DEPR={bool(lyr_DEPR)}, " 
                          f"WSPD={bool(lyr_WSPD)}, WDIR={bool(lyr_WDIR)}")

    # choose T0→T48 in 3h steps (subset if fewer times available)
    t0 = times[0]
    desired = [t0 + pd.Timedelta(hours=h) for h in range(0, TIME_WINDOW_H+1, TIME_STEP_H)]
    chosen = [t for t in desired if t in times]
    if not chosen:  # fallback: stride through whatever is available
        chosen = times[::TIME_STEP_H] or times

    # Pressure levels to fetch
    levels = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]

    # Create layer accessors
    acc_T    = make_level_accessor(caps, lyr_T)
    acc_DEPR = make_level_accessor(caps, lyr_DEPR)
    acc_WS   = make_level_accessor(caps, lyr_WSPD)
    acc_WD   = make_level_accessor(caps, lyr_WDIR)

    # Spatial setup - use model-specific BBOX
    bbox_deg = MODEL_BBOX_DEG.get(model, 0.1)  # Default to 0.1 if model not configured
    half = bbox_deg / 2.0
    bbox = (POINT["lat"]-half, POINT["lon"]-half, POINT["lat"]+half, POINT["lon"]+half)
    w, h = CANVAS_PX

    # Thread pool for concurrent requests (optimized from testing: 24 workers)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=24)

    for t_iso in chosen:
        time_str = t_iso.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # OPTIMIZATION: Try to fetch entire vertical profile in single WCS call
        rows = []
        if USE_VERTICAL_SLICE:
            # Attempt multi-level retrieval for each variable
            T_profile = wcs_get_vertical_profile(lyr_T, time_str, POINT["lat"], POINT["lon"], bbox, (w,h), levels, model)
            DEPR_profile = wcs_get_vertical_profile(lyr_DEPR, time_str, POINT["lat"], POINT["lon"], bbox, (w,h), levels, model)
            WSPD_profile = wcs_get_vertical_profile(lyr_WSPD, time_str, POINT["lat"], POINT["lon"], bbox, (w,h), levels, model)
            WDIR_profile = wcs_get_vertical_profile(lyr_WDIR, time_str, POINT["lat"], POINT["lon"], bbox, (w,h), levels, model)
            
            # If all profiles retrieved successfully, combine them
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
                    
                    rows.append({"pressure_hpa": int(p), "T_C": T_c, "Td_C": Td_c,
                                "wind_dir_deg": wdir, "wind_spd_ms": wspd, "wind_spd_kmh": wspd_kmh})
        
        # FALLBACK: If vertical slice failed or disabled, use individual level requests
        if not rows:
            def fetch_row(p):
                """Fetch data for a single pressure level."""
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

                return {"pressure_hpa": int(p), "T_C": T_c, "Td_C": Td_c,
                        "wind_dir_deg": wdir, "wind_spd_ms": wspd, "wind_spd_kmh": wspd_kmh}

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

        # Cache for overplot
        PROFILE_CACHE[(model, t_iso)] = prof.copy()

        # Inversion detection
        inv = detect_inversion(prof, max_depth_m=2000.0)
        inv_text = "No inversion detected within 2 km AGL."
        warm_req = None
        if inv:
            base_z, top_z, dT = inv
            warm_req = warming_needed_to_break(prof, base_z, top_z)
            inv_text = (f"Inversion: base ~{base_z:.0f} m, top ~{top_z:.0f} m; strength ΔT ≈ {dT:.1f} °C. "
                        f"Estimated surface warming to break to top ≈ {warm_req:.1f} °C.")

        # Mixing height (current), transport wind & VI
        mix_h = current_mixing_height(prof)
        mean_wspd = float(prof.loc[prof["z_m"] <= max(mix_h, 1.0), "wind_spd_ms"].mean())
        mean_wspd_kmh = mean_wspd * 3.6
        VI = ventilation_index(mix_h, mean_wspd)

        # Near-surface crossover temps (RH targets)
        td0 = float(prof.loc[0,"Td_C"])
        t50 = crossover_temperature_from_td(td0, 50.0)   # RH=50%
        t30 = crossover_temperature_from_td(td0, 30.0)   # RH=30%

        # Aloft wind mixing-down hint: check 850 hPa vs warming requirement
        w850 = prof.loc[(prof["pressure_hpa"]==850), "wind_spd_ms"]
        if len(w850):
            w850 = float(w850.iloc[0])
            if warm_req is not None and warm_req <= 3.0 and w850 >= 10.0:  # ~20 kt
                mixdown_note = f"Aloft winds ~850 hPa strong; with ≥{warm_req:.1f} °C warming, momentum mix-down plausible."
            elif warm_req is not None and warm_req > 5.0:
                mixdown_note = "Strong inversion; >5 °C warming likely required before aloft winds mix down."
            else:
                mixdown_note = "Aloft wind mix-down limited by stability."
        else:
            mixdown_note = "850 hPa wind unavailable; mix-down assessment limited."

        winter_note = ("Winter sun angle & high albedo may limit daytime warming and boundary‑layer growth; "
                       "ventilation estimates should be treated conservatively.")

        # Note: File output and plotting handled by parent script

    executor.shutdown(wait=True)
    
    model_elapsed = time.perf_counter() - model_timer_start
    # Return all profiles collected for this model
    return {"profiles": dict(PROFILE_CACHE), "elapsed": model_elapsed}

def main():
    t_start = time.perf_counter()
    print(f"[CONFIG] Models: {MODELS}")
    print(f"[CONFIG] Vertical slice: {USE_VERTICAL_SLICE}, Cache: {ENABLE_CACHE}")
    print(f"[CONFIG] Time window: T+0 to T+{TIME_WINDOW_H} every {TIME_STEP_H}h")
    print(f"[CONFIG] Point: {POINT['lat']:.2f}°N, {POINT['lon']:.2f}°E\n")
    
    # PARALLEL MODEL PROCESSING: Run all models concurrently
    print(f"[PROCESSING] Running {len(MODELS)} models in parallel...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        futures = {executor.submit(run_model, m): m for m in MODELS}
        
        for future in concurrent.futures.as_completed(futures):
            m = futures[future]
            try:
                result = future.result()
                model_elapsed = result["elapsed"]
                print(f"[{m}] Completed in {model_elapsed:.1f} s")
            except Exception as e:
                print(f"[{m}] FAILED: {e}")

    elapsed = time.perf_counter() - t_start
    total_requests = CACHE_HITS + CACHE_MISSES
    cache_hit_rate = (CACHE_HITS / total_requests * 100) if total_requests > 0 else 0
    
    print(f"[STATS] Total API calls: {WMS_CALLS}; elapsed: {elapsed:.1f} s")
    if ENABLE_CACHE:
        print(f"[CACHE] Hits: {CACHE_HITS}, Misses: {CACHE_MISSES}, Hit rate: {cache_hit_rate:.1f}%")
        print(f"[CACHE] Saved {CACHE_HITS} API calls (cache location: {CACHE_DIR})")

if __name__ == "__main__":
    main()
