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
    _load_roads()
    print("Models loaded." if _model is not None else "Running without SkyCLIP model.")

_buildings = None
_roads = None
MAJOR_ROADS = ("motorway", "trunk", "primary", "secondary")

def _load_roads() -> gpd.GeoDataFrame:
    global _roads
    if _roads is None:
        try:
            r = gpd.read_file(OSM_ROADS)
            r = r.set_crs("EPSG:4326") if r.crs is None else r.to_crs("EPSG:4326")
            keep = [c for c in ("highway", "name", "geometry") if c in r.columns]
            _roads = r[keep]
            _roads.sindex
        except Exception as e:
            print(f"[WARNING] Could not load OSM roads from {OSM_ROADS}: {e}")
            return gpd.GeoDataFrame()
    return _roads

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

def _short(text: str, limit: int = 70) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",;") + "…"

def _top_counts(series, n=3) -> str:
    vc = series.dropna().astype(str).value_counts().head(n)
    return ", ".join(f"{c} {v}" for v, c in vc.items())

def _add_tile_facts(tile_polys, top_idx, feats, user_query, parsed, n_total):
    """Adds per-tile explanation columns: percentile, matched_on, building_mix,
    named_places, roads and a one-line `facts` summary."""
    all_emb = _faiss_index.reconstruct_n(0, n_total)  # L2-normalized tile embeddings

    # Combined-score z: how unusual each tile's similarity is within the whole area
    combined = all_emb @ (feats.mean(axis=0) / np.linalg.norm(feats.mean(axis=0))) \
        if feats is not None else None
    # Which prompt drove the match: compare z-scores so the two prompts are on one scale
    if feats is not None:
        sims = all_emb @ feats.T                         # (n_total, 2)
        z = (sims - sims.mean(axis=0)) / (sims.std(axis=0) + 1e-9)

    buildings, roads = _load_buildings(), _load_roads()
    desc = parsed.get("visual_description") or user_query
    rows = {k: [] for k in ("percentile", "matched_on", "building_mix",
                            "named_places", "roads", "facts")}
    for rank, (idx, geom) in enumerate(zip(top_idx, tile_polys.geometry), start=1):
        pct = rank / n_total * 100
        if combined is not None:
            zc = (combined[idx] - combined.mean()) / (combined.std() + 1e-9)
            strength = "very strongly" if zc >= 2.5 else "strongly" if zc >= 1.5 else "moderately"
            on_words = z[idx, 0] >= z[idx, 1]
            matched_on = (f"your wording '{user_query}'" if on_words
                          else f"the visual cue '{_short(desc)}'")
            match_txt = f"imagery matches {strength}, mainly on {matched_on}"
        else:
            matched_on, match_txt = "n/a", "model unavailable (random ranking)"

        b_in = buildings.iloc[buildings.sindex.query(geom, predicate="intersects")] \
            if not buildings.empty else buildings
        mix = _top_counts(b_in["building"]) if "building" in b_in.columns else ""
        names = ", ".join(b_in["name"].dropna().astype(str).unique()[:2]) \
            if "name" in b_in.columns else ""
        r_in = roads.iloc[roads.sindex.query(geom, predicate="intersects")] \
            if not roads.empty else roads
        major = r_in[r_in["highway"].isin(MAJOR_ROADS)] if "highway" in r_in.columns else r_in
        road_names = ", ".join(major["name"].dropna().astype(str).unique()[:2]) \
            if "name" in major.columns else ""

        parts = [f"Ranked #{rank} of {n_total} areas (top {pct:.1f}%)", match_txt,
                 f"contains {len(b_in)} mapped buildings" + (f" ({mix})" if mix else "")]
        if names:
            parts.append(f"includes {names}")
        if road_names:
            parts.append(f"near {road_names}")
        rows["percentile"].append(round(pct, 1))
        rows["matched_on"].append(matched_on)
        rows["building_mix"].append(mix)
        rows["named_places"].append(names)
        rows["roads"].append(road_names)
        rows["facts"].append("; ".join(parts) + ".")
    for k, v in rows.items():
        tile_polys[k] = v

def query_pipeline(user_query: str, top_k: int = TOP_K_DEFAULT, parsed: dict = None) -> gpd.GeoDataFrame:
    return search_with_tiles(user_query, top_k, parsed)[0]

def search_with_tiles(user_query: str, top_k: int = TOP_K_DEFAULT, parsed: dict = None):
    """Returns (results, ranked matched tiles)."""
    if parsed is None:
        parsed = parse_query(user_query)

    # Encode text
    feats = None
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
         
    # Rank every tile (cheap: a few hundred) so results can report percentiles
    n_total = _faiss_index.ntotal
    D_all, I_all = _faiss_index.search(text_feat, n_total)
    D, I = D_all[:, :top_k], I_all[:, :top_k]
    try:
        tiles = json.load(open(TILE_INDEX))
    except FileNotFoundError:
        print("[WARNING] Tile index not found")
        tiles = []

    top_idx = [i for i in I[0] if 0 <= i < len(tiles)]
    top_tiles = [tiles[i] for i in top_idx]
    scores = D[0][:len(top_tiles)].tolist()

    # Build tile polygons GeoDataFrame
    tile_polys = gpd.GeoDataFrame(
        [{"geometry": shapely_box(*t["bbox"]), "tile_id": t["tile_id"], "score": scores[i]}
         for i, t in enumerate(top_tiles)],
        crs="EPSG:4326"
    )
    if not tile_polys.empty:
        _add_tile_facts(tile_polys, top_idx, feats, user_query, parsed, n_total)

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
