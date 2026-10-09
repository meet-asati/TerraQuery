import threading
import faiss
import numpy as np
import json
import geopandas as gpd
import torch
import open_clip
from shapely.geometry import box as shapely_box
from config import *
from parser import parse_query

_model = None
_preprocess = None
_tokenizer = None
_faiss_index = None
_load_lock = threading.Lock()

def load_models():
    # Lock: concurrent Streamlit sessions would otherwise each load the model (RAM exhaustion)
    with _load_lock:
        _load_models()

def _load_models():
    global _model, _preprocess, _tokenizer, _faiss_index
    if _model is not None:
        return
    try:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=SKYCLIP_CHECKPOINT)
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    except Exception as e:
        # open_clip raises RuntimeError (not FileNotFoundError) for a missing checkpoint
        print(f"[WARNING] Could not load SkyCLIP checkpoint ({e}) — text search will be random")
        _model = _preprocess = _tokenizer = None

    try:
        _faiss_index = faiss.read_index(FAISS_INDEX)
    except Exception:
        # Only generate synthetic data when no real index exists, so a real index is never overwritten
        print("[WARNING] FAISS index not found — switching to demo mode")
        from indexer import demo_mode
        demo_mode()
        _faiss_index = faiss.read_index(FAISS_INDEX)
    _load_buildings()  # slow first read (~1 min); do it at startup, not on first query
    print("Models loaded." if _model is not None else "Running without SkyCLIP model.")

_buildings = None

def _load_buildings() -> gpd.GeoDataFrame:
    global _buildings
    if _buildings is None:
        try:
            b = gpd.read_file(OSM_BUILDINGS)
            b = b.set_crs("EPSG:4326") if b.crs is None else b.to_crs("EPSG:4326")
            # Keep only useful columns; the raw OSM export has ~180 sparse tag columns
            keep = [c for c in ("building", "name", "amenity", "landuse", "power",
                                "solar_panel", "geometry") if c in b.columns]
            _buildings = b[keep]
            _buildings.sindex  # build spatial index once
        except Exception as e:
            print(f"[WARNING] Could not load OSM buildings from {OSM_BUILDINGS}: {e}")
            return gpd.GeoDataFrame()
    return _buildings

def query_pipeline(user_query: str, top_k: int = TOP_K_DEFAULT, parsed: dict = None) -> gpd.GeoDataFrame:
    return search_with_tiles(user_query, top_k, parsed)[0]

def search_with_tiles(user_query: str, top_k: int = TOP_K_DEFAULT, parsed: dict = None):
    """Returns (results, ranked matched tiles)."""
    if parsed is None:
        parsed = parse_query(user_query)

    # Encode text
    if _tokenizer is None:
        print("[WARNING] Tokenizer not loaded, cannot encode text")
        text_feat = np.random.randn(1, EMBEDDING_DIM).astype("float32")
    else:
        # SkyCLIP was trained on "a satellite image of ..." captions: combine the user's own
        # words in that template with the LLM's visual description, then average
        prompts = [f"a satellite image of {user_query}",
                   parsed.get("visual_description") or user_query]
        with torch.no_grad():
            feats = _model.encode_text(_tokenizer(prompts)).cpu().numpy().astype("float32")
        faiss.normalize_L2(feats)
        text_feat = feats.mean(axis=0, keepdims=True)

    faiss.normalize_L2(text_feat)

    # FAISS search
    if _faiss_index is None:
         print("[WARNING] FAISS index not loaded, returning empty")
         return gpd.GeoDataFrame(), gpd.GeoDataFrame()
         
    D, I = _faiss_index.search(text_feat, top_k)
    try:
        tiles = json.load(open(TILE_INDEX))
    except FileNotFoundError:
        print("[WARNING] Tile index not found")
        tiles = []
        
    top_tiles = [tiles[i] for i in I[0] if i < len(tiles)]
    scores = D[0][:len(top_tiles)].tolist()

    # Build tile polygons GeoDataFrame
    tile_polys = gpd.GeoDataFrame(
        [{"geometry": shapely_box(*t["bbox"]), "tile_id": t["tile_id"], "score": scores[i]}
         for i, t in enumerate(top_tiles)],
        crs="EPSG:4326"
    )

    # Load (cached after first query) and filter OSM buildings
    buildings = _load_buildings()

    for k, v in (parsed.get("osm_tags") or {}).items():
        if k in buildings.columns:
            # "yes" is OSM's wildcard (building=yes should also match apartments, house, ...)
            if v in ("yes", True):
                buildings = buildings[buildings[k].notna()]
            else:
                buildings = buildings[buildings[k] == v]

    tile_polys["rank"] = range(1, len(tile_polys) + 1)

    # Gap detection: visual tiles with no OSM match
    if parsed.get("task") == "gap_detection":
        if buildings.empty:
            results = tile_polys
        else:
            joined = gpd.sjoin(tile_polys, buildings, how="left", predicate="intersects")
            results = joined[joined["index_right"].isna()].copy()
    # Normal search: OSM features inside matching tiles
    elif buildings.empty or tile_polys.empty:
        results = tile_polys  # fallback: return tile matches
    else:
        results = gpd.sjoin(buildings, tile_polys, how="inner", predicate="intersects")
        results = results.sort_values("score", ascending=False)

    # Ranked matched tiles, for the UI to highlight what the visual search actually found
    counts = results["tile_id"].value_counts() if "tile_id" in results.columns else {}
    tile_polys["n_features"] = [int(counts.get(t, 0)) for t in tile_polys["tile_id"]]
    return results, tile_polys

def get_output_formats(results: gpd.GeoDataFrame) -> dict:
    if results.empty:
        return {"geojson": "{}", "centroids": [], "bboxes": [], "count": 0}
        
    return {
        "geojson": results.to_json(),
        "centroids": [[r.geometry.centroid.y, r.geometry.centroid.x]
                      for _, r in results.iterrows()],
        "bboxes": [list(r.geometry.bounds) for _, r in results.iterrows()],
        "count": len(results)
    }

__all__ = ["load_models", "query_pipeline", "search_with_tiles", "get_output_formats"]
