import requests
import math
import json
import os
import time
import torch
import open_clip
import faiss
import numpy as np
from PIL import Image
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

from config import *

def deg2tile(lat, lon, zoom):
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_r) + 1/math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y

def tile2bbox(x, y, zoom):
    n = 2 ** zoom
    lon_min = x / n * 360 - 180
    lon_max = (x + 1) / n * 360 - 180
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*y/n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*(y+1)/n))))
    return [lon_min, lat_min, lon_max, lat_max]

def fetch_tile(x, y, zoom):
    # Try primary then fallback server; reject non-image responses (error/HTML pages)
    for server in (TILE_SERVER, TILE_SERVER_FALLBACK):
        url = server.format(x=x, y=y, z=zoom)
        try:
            r = requests.get(url, headers={"User-Agent":"TerraQuery-AI/1.0"}, timeout=10)
            r.raise_for_status()
            if r.headers.get("Content-Type", "").startswith("image/"):
                return r.content
        except requests.RequestException:
            pass
    return None

def fetch_tiles_for_aoi(bbox=AOI_BBOX, zoom=ZOOM_LEVEL):
    x_min, y_min = deg2tile(bbox[0], bbox[3], zoom)  # north, west
    x_max, y_max = deg2tile(bbox[1], bbox[2], zoom)  # south, east
    tile_records = []
    os.makedirs(TILE_DIR, exist_ok=True)
    
    total = (x_max - x_min + 1) * (y_max - y_min + 1)
    count = 0
    
    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            tile_id = f"tile_{zoom}_{x}_{y}"
            path = f"{TILE_DIR}/{tile_id}.png"
            count += 1

            content = fetch_tile(x, y, zoom)
            if content is None:
                print(f"[WARNING] Skipping {tile_id}: all tile servers unreachable")
                continue
            with open(path, "wb") as f:
                f.write(content)

            tile_records.append({"tile_id": tile_id, "path": path, "bbox": tile2bbox(x, y, zoom)})
            time.sleep(0.05)
            if count % 50 == 0:
                print(f"Downloaded {count}/{total} tiles...")
                
    json.dump(tile_records, open(TILE_INDEX, "w"))
    return len(tile_records)

def build_faiss_index():
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=SKYCLIP_CHECKPOINT)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tiles = json.load(open(TILE_INDEX))
    embeddings = []
    
    count = 0
    total = len(tiles)
    
    with torch.no_grad():
        for tile in tiles:
            img = preprocess(Image.open(tile["path"]).convert("RGB")).unsqueeze(0).to(device)
            feat = model.encode_image(img).squeeze().cpu().numpy()
            embeddings.append(feat)
            count += 1
            if count % 50 == 0:
                print(f"Indexed {count}/{total} tiles...")
                
    if not embeddings:
        print("No embeddings generated, skipping FAISS index creation.")
        return 0
        
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX)
    return len(embeddings)

OVERPASS_ENDPOINTS = ["https://overpass-api.de/api", "https://overpass.kumi.systems/api"]

def download_osm_data(bbox=AOI_BBOX):
    buildings = roads = None
    for endpoint in OVERPASS_ENDPOINTS:
        ox.settings.overpass_endpoint = endpoint
        try:
            buildings = ox.features_from_bbox(*bbox, tags={"building": True})
            roads = ox.features_from_bbox(*bbox, tags={"highway": True})
            break
        except Exception as e:
            print(f"[WARNING] Overpass endpoint {endpoint} failed: {e}")
    if buildings is None or roads is None:
        if os.path.exists(OSM_BUILDINGS) and os.path.exists(OSM_ROADS):
            print("[WARNING] All Overpass endpoints unreachable — keeping existing OSM data")
            return
        raise RuntimeError("All Overpass endpoints unreachable and no cached OSM data found")
    buildings.to_file(OSM_BUILDINGS, driver="GPKG")
    roads.to_file(OSM_ROADS, driver="GPKG")
    print(f"Downloaded {len(buildings)} buildings and {len(roads)} roads")
    download_osm_pois(bbox)

# Points of interest that are often mapped without a building tag (hospital nodes, stations, parks...)
POI_TAGS = {"amenity": True, "healthcare": True, "leisure": True, "tourism": True, "shop": True,
            "office": True, "historic": True, "railway": ["station", "halt"],
            "public_transport": "station"}
POI_COLUMNS = ("name", "name:en", "alt_name", "short_name", "official_name", "old_name",
               "amenity", "healthcare", "leisure", "tourism", "shop", "office", "historic",
               "railway", "public_transport", "religion", "building", "sport", "operator",
               # station type: separates suburban/main-line trains from metro and monorail
               "station", "train", "subway", "monorail", "light_rail", "network", "usage")

def download_osm_pois(bbox=AOI_BBOX):
    for endpoint in OVERPASS_ENDPOINTS:
        ox.settings.overpass_endpoint = endpoint
        try:
            pois = ox.features_from_bbox(*bbox, tags=POI_TAGS)
            break
        except Exception as e:
            print(f"[WARNING] Overpass endpoint {endpoint} failed for POIs: {e}")
    else:
        print("[WARNING] POI download failed — keeping existing POI data (if any)")
        return
    pois = pois.reset_index()  # element_type / osmid become columns
    keep = [c for c in ("element_type", "osmid", *POI_COLUMNS) if c in pois.columns]
    pois = gpd.GeoDataFrame(pois[keep + ["geometry"]], geometry="geometry", crs=pois.crs)
    pois.to_file(OSM_POIS, driver="GPKG")
    print(f"Downloaded {len(pois)} points of interest")
    download_osm_landcover(bbox)

# Water bodies and land cover, used as spatial-relation targets ("within 200 m of water bodies")
LANDCOVER_TAGS = {"natural": True, "waterway": True, "water": True, "landuse": True}
LANDCOVER_COLUMNS = ("name", "name:en", "natural", "waterway", "water", "landuse", "leisure")

def download_osm_landcover(bbox=AOI_BBOX):
    for endpoint in OVERPASS_ENDPOINTS:
        ox.settings.overpass_endpoint = endpoint
        try:
            lc = ox.features_from_bbox(*bbox, tags=LANDCOVER_TAGS)
            break
        except Exception as e:
            print(f"[WARNING] Overpass endpoint {endpoint} failed for land cover: {e}")
    else:
        print("[WARNING] Land-cover download failed — keeping existing data (if any)")
        return
    lc = lc.reset_index()
    keep = [c for c in ("element_type", "osmid", *LANDCOVER_COLUMNS) if c in lc.columns]
    lc = gpd.GeoDataFrame(lc[keep + ["geometry"]], geometry="geometry", crs=lc.crs)
    lc.to_file(OSM_LANDCOVER, driver="GPKG")
    print(f"Downloaded {len(lc)} land-cover / water features")

def demo_mode():
    os.makedirs(TILE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FAISS_INDEX), exist_ok=True)
    
    # 1. Synthetic tiles
    tile_records = []
    for i in range(10):
        # random small bboxes around AOI_CENTER
        lat = AOI_CENTER[0] + np.random.uniform(-0.01, 0.01)
        lon = AOI_CENTER[1] + np.random.uniform(-0.01, 0.01)
        bbox = [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001]
        tile_records.append({
            "tile_id": f"demo_tile_{i}",
            "path": f"{TILE_DIR}/demo_{i}.png",
            "bbox": bbox
        })
        # create dummy image
        Image.new("RGB", (TILE_SIZE, TILE_SIZE), color=(73, 109, 137)).save(f"{TILE_DIR}/demo_{i}.png")
        
    json.dump(tile_records, open(TILE_INDEX, "w"))
    
    # 2. Synthetic FAISS index
    embeddings = np.random.randn(10, EMBEDDING_DIM).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX)
    
    # 3. Synthetic GeoDataFrame (Buildings)
    points = [Point(lon + np.random.uniform(-0.005, 0.005), lat + np.random.uniform(-0.005, 0.005)) for i in range(5)]
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    gdf["building"] = "yes"
    gdf.to_file(OSM_BUILDINGS, driver="GPKG")
    
    # synthetic roads
    gdf_roads = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    gdf_roads["highway"] = "residential"
    gdf_roads.to_file(OSM_ROADS, driver="GPKG")
    
    print("[DEMO MODE] Synthetic data generated. Run indexer.py for real results.")

if __name__ == "__main__":
    print("Step 1/3: Fetching satellite tiles for Kurla-Bandra, Mumbai...")
    n = fetch_tiles_for_aoi()
    print(f"Downloaded {n} tiles")
    print("Step 2/3: Building FAISS index with SkyCLIP...")
    n = build_faiss_index()
    print(f"Indexed {n} tiles")
    print("Step 3/3: Downloading OSM data...")
    download_osm_data()
    print("All done. Run: streamlit run app.py")
