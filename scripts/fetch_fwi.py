import os
import json
import datetime
import requests
from bs4 import BeautifulSoup
import geopandas as gpd
from shapely.geometry import Point, box
import pyproj
import numpy as np
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed

GRID_STEP = 0.25
BBOX = [44.0, 25.0, 63.5, 40.0]
WMS_URL = "https://maps.effis.emergency.copernicus.eu/gwis"

# تبدیل مختصات جغرافیایی به مرکاتور
transformer_to_3857 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

def lonlat_to_mercator(lon, lat):
    return transformer_to_3857(lon, lat)

# بارگذاری مرز دقیق کشور ایران
iran_geom = None
geojson_path = "IRAN.geojson"
if os.path.exists(geojson_path):
    try:
        gdf = gpd.read_file(geojson_path)
        iran_geom = gdf.unary_union
        print(f"Loaded {geojson_path} successfully.")
    except Exception as e:
        print(f"Error loading {geojson_path}: {e}")

if iran_geom is None or iran_geom.is_empty:
    print("Fallback: Using Iran BBOX.")
    iran_geom = box(BBOX[0], BBOX[1], BBOX[2], BBOX[3])

# ایجاد شبکه نقاط درون خاک ایران
lons = np.arange(BBOX[0], BBOX[2] + GRID_STEP, GRID_STEP)
lats = np.arange(BBOX[1], BBOX[3] + GRID_STEP, GRID_STEP)
target_points = []

for lat in lats:
    for lon in lons:
        p = Point(lon, lat)
        if iran_geom.contains(p):
            target_points.append((round(float(lon), 4), round(float(lat), 4)))

print(f"Total target points in Iran: {len(target_points)}")

today_str = datetime.date.today().strftime("%Y-%m-%d")

def fetch_single_point(coords):
    lon, lat = coords
    mx, my = lonlat_to_mercator(lon, lat)
    delta = 2500  # فاصله در سیستم مرکاتور بر حسب متر
    bbox_str = f"{mx - delta},{my - delta},{mx + delta},{my + delta}"

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": "ecmwf.query",
        "QUERY_LAYERS": "ecmwf.query",
        "BBOX": bbox_str,
        "WIDTH": "101",
        "HEIGHT": "101",
        "X": "50",
        "Y": "50",
        "SRS": "EPSG:3857",
        "INFO_FORMAT": "text/html",
        "TIME": today_str
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    val = None
    try:
        resp = requests.get(WMS_URL, params=params, headers=headers, timeout=10)
        if resp.status_code == 200 and "fwi" in resp.text.lower():
            soup = BeautifulSoup(resp.text, "lxml")
            cells = soup.find_all(["td", "th"])
            for i, c in enumerate(cells):
                if "fwi" in c.get_text().lower() and i + 1 < len(cells):
                    txt = cells[i + 1].get_text().strip()
                    try:
                        flt = float(txt)
                        if 0 <= flt <= 150:
                            val = flt
                            break
                    except ValueError:
                        pass
            if val is None:
                for c in cells:
                    txt = c.get_text().strip()
                    try:
                        flt = float(txt)
                        if 0 <= flt <= 150:
                            val = flt
                            break
                    except ValueError:
                        pass
    except Exception:
        pass

    return lon, lat, val

fetched_data = []
missing_points = []

# دریافت موازی داده‌ها با ۱۰ پردازش همزمان
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_single_point, pt) for pt in target_points]
    for future in as_completed(futures):
        lon, lat, val = future.result()
        if val is not None:
            fetched_data.append({
                "lon": lon,
                "lat": lat,
                "fwi": round(float(val), 2),
                "interpolated": False
            })
        else:
            missing_points.append((lon, lat))

print(f"Direct points fetched: {len(fetched_data)}")
print(f"Missing points: {len(missing_points)}")

# درون‌یابی مکانی (IDW) برای نقاط خالی
if fetched_data and missing_points:
    known_coords = np.array([[p["lon"], p["lat"]] for p in fetched_data])
    known_vals = np.array([p["fwi"] for p in fetched_data])
    tree = cKDTree(known_coords)

    for lon, lat in missing_points:
        k = min(8, len(known_coords))
        distances, indices = tree.query([lon, lat], k=k)

        if np.any(distances < 1e-5):
            interp_val = known_vals[indices[np.argmin(distances)]]
        else:
            weights = 1.0 / (distances ** 2)
            interp_val = np.sum(weights * known_vals[indices]) / np.sum(weights)

        fetched_data.append({
            "lon": lon,
            "lat": lat,
            "fwi": round(float(interp_val), 2),
            "interpolated": True
        })

all_fwi = [p["fwi"] for p in fetched_data] if fetched_data else [0.0]

output_data = {
    "forecast_date": today_str,
    "statistics": {
        "count": len(fetched_data),
        "mean": round(float(np.mean(all_fwi)), 2),
        "max": round(float(np.max(all_fwi)), 2),
        "min": round(float(np.min(all_fwi)), 2)
    },
    "points": fetched_data
}

os.makedirs("data", exist_ok=True)
with open("data/fwi_fars.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"Successfully saved data/fwi_fars.json with {len(fetched_data)} points.")
