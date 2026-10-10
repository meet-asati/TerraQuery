"""Embed a SkyCLIP image crop around every OSM building into a per-building FAISS index.

Run once after indexer.py (uses its tiles + OSM buildings):
    python build_building_index.py
Uses the GPU when torch has CUDA (fp16), otherwise the CPU (slower).
"""
import json
import math
import time

import faiss
import geopandas as gpd
import numpy as np
import open_clip
import torch
from PIL import Image

from config import (BUILDING_FAISS, BUILDING_ROWS, CLIP_MODEL, EMBEDDING_DIM, OSM_BUILDINGS,
                    SKYCLIP_CHECKPOINT, TILE_INDEX, TILE_SIZE, ZOOM_LEVEL)

CROP_SCALE = 1.6      # crop = footprint size x this, so the roof is seen with a little context
MIN_CROP_PX = 48      # ~55 m at z17: tiny buildings are unreadable without surroundings
MAX_CROP_PX = 224
BATCH = 128


def lonlat_to_px(lon, lat, x0, y0, zoom=ZOOM_LEVEL):
    """Web-Mercator lon/lat -> pixel position in the stitched mosaic (origin = tile x0, y0)."""
    n = 2 ** zoom
    lat_r = math.radians(lat)
    tx = (lon + 180) / 360 * n
    ty = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
    return (tx - x0) * TILE_SIZE, (ty - y0) * TILE_SIZE


def build_mosaic(tiles):
    xy = [tuple(map(int, t["tile_id"].split("_")[2:4])) for t in tiles]
    x0, y0 = min(x for x, _ in xy), min(y for _, y in xy)
    nx, ny = max(x for x, _ in xy) - x0 + 1, max(y for _, y in xy) - y0 + 1
    mosaic = Image.new("RGB", (nx * TILE_SIZE, ny * TILE_SIZE))
    for t, (x, y) in zip(tiles, xy):
        img = Image.open(t["path"]).convert("RGB").resize((TILE_SIZE, TILE_SIZE))
        mosaic.paste(img, ((x - x0) * TILE_SIZE, (y - y0) * TILE_SIZE))
    return mosaic, x0, y0


def main():
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=SKYCLIP_CHECKPOINT)
    model = model.eval().to(device)

    tiles = json.load(open(TILE_INDEX))
    mosaic, x0, y0 = build_mosaic(tiles)
    W, H = mosaic.size
    print(f"Mosaic: {len(tiles)} tiles -> {W}x{H}px")

    buildings = gpd.read_file(OSM_BUILDINGS)
    buildings = buildings.set_crs("EPSG:4326") if buildings.crs is None else buildings.to_crs("EPSG:4326")
    print(f"Buildings: {len(buildings)}")

    rows, crops = [], []
    for pos, geom in enumerate(buildings.geometry):
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        px0, py0 = lonlat_to_px(minx, maxy, x0, y0)   # top-left
        px1, py1 = lonlat_to_px(maxx, miny, x0, y0)   # bottom-right
        cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
        if not (0 <= cx < W and 0 <= cy < H):
            continue  # outside imagery coverage
        size = min(max(max(px1 - px0, py1 - py0) * CROP_SCALE, MIN_CROP_PX), MAX_CROP_PX)
        half = size / 2
        crops.append((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
        rows.append(pos)
    print(f"Crops: {len(crops)} ({len(buildings) - len(crops)} buildings outside imagery skipped)")

    feats = np.zeros((len(crops), EMBEDDING_DIM), dtype="float32")
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        for start in range(0, len(crops), BATCH):
            batch = torch.stack([preprocess(mosaic.crop(b)) for b in crops[start:start + BATCH]]).to(device)
            feats[start:start + len(batch)] = model.encode_image(batch).float().cpu().numpy()
            done = start + len(batch)
            if done % (BATCH * 20) < BATCH or done == len(crops):
                print(f"Embedded {done}/{len(crops)}")

    faiss.normalize_L2(feats)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(feats)
    faiss.write_index(index, BUILDING_FAISS)
    np.save(BUILDING_ROWS, np.array(rows, dtype="int64"))
    print(f"Saved {index.ntotal} building embeddings to {BUILDING_FAISS} in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
