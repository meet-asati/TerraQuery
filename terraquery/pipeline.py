import math
import threading
import faiss
import numpy as np
import pandas as pd
import json
import geopandas as gpd
import torch
import open_clip
from shapely.geometry import Point, box as shapely_box
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
    _load_pois()
    _load_landcover()
    _load_infra()
    _load_building_index()
    print("Models loaded." if _model is not None else "Running without SkyCLIP model.")

_buildings = None
_roads = None
_pois = None
CATEGORY_LIMIT = 50          # max features listed for a category query ("hospitals")
QUALIFIER_KEYS = {"religion", "denomination", "sport", "cuisine", "operator", "station"}
METRIC_CRS = "EPSG:32643"     # UTM 43N (Mumbai): distances in metres
DEFAULT_NEAR_M = 500
WATER_KEYS = {"natural", "water", "waterway", "name~"}   # tags that need the land-cover layer
# tags that need the railway/power layer (railway=station/halt stay in the POI layer)
POWER_KEYS = {"power", "generator:source", "generator:method", "plant:source"}
STATION_VALUES = {"station", "halt"}
_landcover = None
_infra = None
# Per-building embeddings (build_building_index.py): FAISS index, embedding->building row,
# building row->embedding, and the raw normalized vectors for whole-area statistics
_bld_index = _bld_rows = _row_to_emb = _bld_emb = None
# OSM name variants searched for named queries (excludes e.g. name:etymology:wikidata)
NAME_COLS = ("name", "name:en", "alt_name", "short_name", "official_name",
             "old_name", "int_name", "loc_name", "addr:housename")
_NAME_STOPWORDS = {"the", "of", "and", "college", "school", "road", "marg", "station",
                   "building", "near", "mumbai"}
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

def _load_pois() -> gpd.GeoDataFrame:
    global _pois
    if _pois is None:
        try:
            p = gpd.read_file(OSM_POIS)
            _pois = p.set_crs("EPSG:4326") if p.crs is None else p.to_crs("EPSG:4326")
            _pois.sindex
        except Exception as e:
            print(f"[WARNING] Could not load OSM POIs from {OSM_POIS} ({e}) - run "
                  "`python -c \"from indexer import download_osm_pois; download_osm_pois()\"`")
            return gpd.GeoDataFrame()
    return _pois

def _as_list(v):
    return v if isinstance(v, (list, tuple, set)) else [v]

def _tag_match(gdf: gpd.GeoDataFrame, tags: dict):
    """OSM tag match. Primary keys are alternatives (OR); qualifier keys (religion...) narrow
    them. Returns (mask, matched_tag) where matched_tag is e.g. 'amenity=hospital'."""
    mask = pd.Series(False, index=gdf.index)
    matched = pd.Series("", index=gdf.index, dtype="object")
    by_building = pd.Series(False, index=gdf.index)
    for k, v in tags.items():
        if k == "name~":
            # Name substrings as one more alternative: "creek" or "talao" have no OSM tag of
            # their own (Danda Creek is water=river, many ponds carry only a name)
            name_cols = [c for c in NAME_COLS if c in gdf.columns]
            if not name_cols:
                continue
            names = gdf[name_cols].astype("string").apply(lambda c: c.str.lower())
            # Only water features: "Lake View Apartments" / "Tank Road" are not water bodies
            is_water = pd.Series(False, index=gdf.index)
            for wk in ("water", "waterway"):
                if wk in gdf.columns:
                    is_water |= gdf[wk].notna()
            if "natural" in gdf.columns:
                is_water |= gdf["natural"].isin(["water", "wetland", "bay", "coastline"])
            for sub in (str(x).lower() for x in _as_list(v)):
                hit = is_water & names.apply(
                    lambda c: c.str.contains(sub, regex=False, na=False)).any(axis=1)
                new = hit & ~mask
                matched[new] = "name~" + sub
                mask |= hit
            continue
        if k in QUALIFIER_KEYS or k not in gdf.columns:
            continue
        col = gdf[k]
        hit = col.notna() if any(x in ("*", "yes", True) for x in _as_list(v)) \
            else col.isin([str(x) for x in _as_list(v)])
        new = hit & ~mask
        matched[new] = k + "=" + col[new].astype(str)
        if k == "building":
            by_building |= new
        mask |= hit
    for k, v in tags.items():
        if k not in QUALIFIER_KEYS:
            continue
        col = gdf[k] if k in gdf.columns else pd.Series(None, index=gdf.index, dtype="object")
        vals = [str(x) for x in _as_list(v)]
        excl = [x[1:] for x in vals if x.startswith("!")]
        incl = [x for x in vals if not x.startswith("!")]
        if incl:   # e.g. religion=hindu (untagged rows pass only if matched via building=temple)
            mask &= col.isin(incl) | (col.isna() & by_building)
        if excl:   # e.g. station=!subway: metro/monorail stops are not railway stations
            mask &= ~col.isin(excl)
    return mask, matched

def _dedupe(results: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Same named place mapped twice (POI node + building polygon): keep the polygon."""
    if results.empty or "name" not in results.columns:
        return results
    cen = results.geometry.representative_point()
    key = results["name"].astype("string").str.lower().str.strip() + "|" + \
        cen.x.round(3).astype(str) + "|" + cen.y.round(3).astype(str)
    is_point = (results.geometry.geom_type == "Point").astype(int)
    order = results.assign(_k=key.fillna(pd.Series(results.index.astype(str), index=results.index)),
                           _p=is_point).sort_values("_p")
    return order.drop_duplicates("_k").drop(columns=["_k", "_p"])

def _dissolve_roads(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge road segments by name so 'main roads' lists roads, not hundreds of segments."""
    lines = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    points = roads[~roads.index.isin(lines.index)]
    named = lines[lines["name"].notna()] if "name" in lines.columns else lines.iloc[0:0]
    if len(named):
        agg = {c: "first" for c in ("highway", "matched_tag", "layer") if c in named.columns}
        named = named.dissolve(by="name", aggfunc=agg, as_index=False)
    merged = pd.concat([named, points], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    merged["_len"] = merged.to_crs(3857).length
    merged["_named"] = merged["name"].notna() if "name" in merged.columns else False
    return merged.sort_values(["_named", "_len"], ascending=False).drop(columns=["_len", "_named"])

def _load_infra() -> gpd.GeoDataFrame:
    global _infra
    if _infra is None:
        try:
            g = gpd.read_file(OSM_INFRA)
            _infra = g.set_crs("EPSG:4326") if g.crs is None else g.to_crs("EPSG:4326")
        except Exception as e:
            print(f"[WARNING] Could not load railway/power data from {OSM_INFRA} ({e}) - run "
                  "`python -c \"from indexer import download_osm_infrastructure; "
                  "download_osm_infrastructure()\"`")
            return gpd.GeoDataFrame()
    return _infra

def _needs_infra(tags: dict) -> bool:
    if any(k in POWER_KEYS for k in tags):
        return True
    rail = tags.get("railway")
    return rail is not None and not set(map(str, _as_list(rail))) <= STATION_VALUES

def _merge_lines(results: gpd.GeoDataFrame, layers=("landcover", "infra")) -> gpd.GeoDataFrame:
    """Features mapped in pieces (river segments, track segments): one result per named
    feature; unnamed track pieces grouped by type ('Unnamed rail (yard)')."""
    sel = results["layer"].isin(layers)
    named = results[sel & results["name"].notna()]
    unnamed_lines = results[sel & results["name"].isna() & results.get(
        "railway", pd.Series(index=results.index, dtype="object")).notna()
        & results.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    pieces = []
    if len(named) > 1:
        agg = {c: "first" for c in named.columns if c not in ("geometry", "name")}
        pieces.append(named.dissolve(by="name", aggfunc=agg, as_index=False))
    else:
        pieces.append(named)
    if len(unnamed_lines):
        u = unnamed_lines.copy()
        svc = u["service"] if "service" in u.columns else pd.Series(None, index=u.index)
        u["group_label"] = ("Unnamed " + u["railway"].astype(str).str.replace("_", " ")
                            + svc.map(lambda x: f" ({x})" if isinstance(x, str) else ""))
        agg = {c: "first" for c in u.columns if c not in ("geometry", "group_label")}
        pieces.append(u.dissolve(by="group_label", aggfunc=agg, as_index=False))
    rest = results.drop(index=named.index.union(unnamed_lines.index))
    merged = gpd.GeoDataFrame(pd.concat([rest, *pieces], ignore_index=True),
                              geometry="geometry", crs="EPSG:4326")
    # Longest first among line features so main lines lead the list
    merged["_len"] = merged.to_crs(METRIC_CRS).length
    return merged

def _load_landcover() -> gpd.GeoDataFrame:
    global _landcover
    if _landcover is None:
        try:
            lc = gpd.read_file(OSM_LANDCOVER)
            _landcover = lc.set_crs("EPSG:4326") if lc.crs is None else lc.to_crs("EPSG:4326")
        except Exception as e:
            print(f"[WARNING] Could not load land cover from {OSM_LANDCOVER} ({e}) - run "
                  "`python -c \"from indexer import download_osm_landcover; download_osm_landcover()\"`")
            return gpd.GeoDataFrame()
    return _landcover

def _find_targets(tags: dict) -> gpd.GeoDataFrame:
    """Every OSM feature (any layer) carrying the target tags of a spatial relation."""
    parts = []
    for layer, gdf in (("building", _load_buildings()), ("poi", _load_pois()),
                       ("road", _load_roads()), ("landcover", _load_landcover()),
                       ("infra", _load_infra())):
        if gdf.empty:
            continue
        mask, matched = _tag_match(gdf, tags)
        if mask.any():
            hit = gdf.loc[mask, ["geometry"]].copy()
            hit["target_name"] = gdf.loc[mask, "name"] if "name" in gdf.columns else None
            hit["target_tag"] = matched[mask]
            parts.append(hit)
    if not parts:
        return gpd.GeoDataFrame(columns=["geometry", "target_name", "target_tag"],
                                geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry",
                            crs="EPSG:4326")

def _spatial_relation(parsed: dict):
    rel = parsed.get("spatial_relation")
    if not isinstance(rel, dict) or not rel.get("target_osm_tags"):
        return None
    try:
        dist = float(rel.get("distance_m") or DEFAULT_NEAR_M)
    except (TypeError, ValueError):
        dist = DEFAULT_NEAR_M
    return {"type": "beyond" if rel.get("type") in ("beyond", "far", "far_from") else "within",
            "distance_m": dist, "tags": rel["target_osm_tags"],
            "label": rel.get("target") or "the target features"}

def _apply_spatial(results: gpd.GeoDataFrame, rel: dict):
    """Keep results within / beyond rel['distance_m'] of the target features (metres, UTM).
    Adds dist_m, target_name, target_tag. Returns (filtered, note, targets_used)."""
    targets = _find_targets(rel["tags"])
    label, dist = rel["label"], rel["distance_m"]
    if targets.empty:
        return results.iloc[0:0], f"No {label} found in the OpenStreetMap data, so nothing " \
            f"can be within {dist:.0f} m of them.", targets
    if results.empty:
        return results, "", targets.iloc[0:0]
    r_m = results[["geometry"]].to_crs(METRIC_CRS)
    t_m = targets.to_crs(METRIC_CRS)
    near = gpd.sjoin_nearest(r_m, t_m, how="left", distance_col="dist_m")
    near = near[~near.index.duplicated(keep="first")]          # equidistant ties
    keep = near["dist_m"] <= dist if rel["type"] == "within" else near["dist_m"] > dist
    out = results.loc[near.index[keep]].copy()
    out["dist_m"] = near.loc[keep, "dist_m"].round().astype(int).values
    out["target_name"] = near.loc[keep, "target_name"].values
    out["target_tag"] = near.loc[keep, "target_tag"].values
    rel_txt = f"within {dist:.0f} m of" if rel["type"] == "within" else f"more than {dist:.0f} m from"
    note = (f"{len(out)} of {len(results)} {rel_txt} {label}" if len(out) else
            f"None of the {len(results)} matching features is {rel_txt} {label}.")
    used = targets.loc[sorted(set(near.loc[keep, "index_right"].dropna().astype(int)))]
    return out, note, used

def _spatial_text(f) -> str:
    if pd.isna(f.get("dist_m")):
        return ""
    tgt = f.get("target_name") if isinstance(f.get("target_name"), str) else None
    what = f"'{tgt}' ({f.get('target_tag')})" if tgt else f"a feature tagged {f.get('target_tag')}"
    return f"{int(f['dist_m'])} m from {what}"

def osm_category_search(user_query, parsed, text_feat, tile_scores, top_k=CATEGORY_LIMIT):
    """Category query ("hospitals", "parks", "main roads"): every OSM feature carrying the
    tags, with exact geometry and names. Imagery similarity only orders the results."""
    tags = parsed.get("osm_tags") or {}
    is_road = any(k == "highway" for k in tags)
    parts = []
    layers = (("road", _load_roads()),) if is_road else \
        (("building", _load_buildings()), ("poi", _load_pois()))
    if not is_road and any(k in WATER_KEYS for k in tags):
        # "show rivers" / "ponds": water lives in the land-cover layer; other categories
        # keep searching buildings + POIs only
        layers = layers + (("landcover", _load_landcover()),)
    if not is_road and _needs_infra(tags):
        # railway tracks / power: line and generator features live in the infra layer
        layers = layers + (("infra", _load_infra()),)
    for layer, gdf in layers:
        if gdf.empty:
            continue
        mask, matched = _tag_match(gdf, tags)
        if mask.any():
            hit = gdf[mask].copy()
            hit["matched_tag"] = matched[mask]
            hit["layer"] = layer
            if layer == "building" and _bld_index is not None:
                hit["score"] = _bld_emb[_row_to_emb[hit.index]] @ text_feat[0]
            parts.append(hit)
    if not parts:
        empty = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
        empty.attrs["no_tag_match"] = True
        return empty
    results = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry",
                               crs="EPSG:4326")
    rel = _spatial_relation(parsed)
    if is_road:
        results = _dissolve_roads(results)
    else:
        # Rivers / tracks mapped in pieces: one result per named feature
        has_lines = results["layer"].isin(["landcover", "infra"]).any()
        if has_lines:
            results = _merge_lines(results)
        results = _dedupe(results)
        # Points/POIs without a crop embedding: use the score of the tile they sit in
        cen = results.geometry.representative_point()
        tile_s = pd.Series([tile_scores.get(_tile_id_at(c.x, c.y), np.nan) for c in cen],
                           index=results.index)
        results["score"] = results["score"].fillna(tile_s) if "score" in results.columns else tile_s
        if has_lines and "_len" in results.columns:   # tracks/rivers: named, then longest
            results = results.assign(_named=results["name"].notna()).sort_values(
                ["_named", "_len"], ascending=[False, False]).drop(columns=["_named", "_len"])
        else:
            results = results.assign(_named=results["name"].notna()).sort_values(
                ["_named", "score"], ascending=[False, False]).drop(columns="_named")
    note, targets = "", None
    if rel:
        results, note, targets = _apply_spatial(results, rel)
        results = results.sort_values("dist_m", ascending=rel["type"] == "within", kind="stable")
    total = len(results)
    results = results.head(top_k).copy()
    if results.empty:
        results.attrs["note"] = note
        return results

    roads = _load_roads()
    reasons, facts = [], []
    for _, f in results.iterrows():
        tag = f.get("matched_tag") or ""
        name = f.get("name") if isinstance(f.get("name"), str) else None
        sp = _spatial_text(f) if rel else ""
        reasons.append(f"Tagged {tag} in OpenStreetMap" + (f" as '{name}'" if name else "")
                       + (f", {sp}" if sp else "") + f", which matches '{user_query}'.")
        bits = [f"OSM tag {tag}", f"{f.geometry.geom_type.lower()} from the {f.get('layer')} layer"]
        if sp:
            bits.append(sp)
        if not is_road and not roads.empty:
            near = roads.iloc[roads.sindex.query(f.geometry.buffer(0.0015), predicate="intersects")]
            major = near[near["highway"].isin(MAJOR_ROADS)]
            rn = ", ".join(major["name"].dropna().astype(str).unique()[:2])
            if rn:
                bits.append(f"near {rn}")
        if pd.notna(f.get("score")):
            bits.append(f"imagery similarity {float(f['score']):.3f}")
        facts.append("; ".join(bits) + ".")
    results["rank"] = range(1, len(results) + 1)
    results["reason"] = reasons
    results["facts"] = facts
    results["source"] = "osm_tags"
    results["total_matches"] = total
    if "score" not in results.columns:
        results["score"] = np.nan
    results.attrs["note"] = note
    if targets is not None and len(targets):
        results.attrs["targets"] = targets   # popped by the UI to draw the target features
    return results

def _load_building_index():
    global _bld_index, _bld_rows, _row_to_emb, _bld_emb
    if _bld_index is not None:
        return
    try:
        index, rows = faiss.read_index(BUILDING_FAISS), np.load(BUILDING_ROWS)
        n_buildings = len(_load_buildings())
        if len(rows) != index.ntotal or (len(rows) and rows.max() >= n_buildings):
            raise ValueError("out of date with the OSM buildings file")
        row_to_emb = np.full(n_buildings, -1, dtype="int64")
        row_to_emb[rows] = np.arange(len(rows))
        _bld_emb = index.reconstruct_n(0, index.ntotal)
        _bld_index, _bld_rows, _row_to_emb = index, rows, row_to_emb
    except Exception as e:
        print(f"[WARNING] Per-building index unavailable ({e}) - run build_building_index.py; "
              "falling back to tile-level results")

def _tile_id_at(lon: float, lat: float) -> str:
    n = 2 ** ZOOM_LEVEL
    x = int((lon + 180) / 360 * n)
    lat_r = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return f"tile_{ZOOM_LEVEL}_{x}_{y}"

def _feature_search(user_query, parsed, text_feat, feats, top_k, tile_rank):
    """Rank individual OSM buildings by their own image-crop similarity to the query.
    Returns one row per matched building (exact OSM footprint) with explanation facts."""
    buildings, roads = _load_buildings(), _load_roads()
    combined = _bld_emb @ text_feat[0]                       # similarity of every building
    order = np.argsort(-combined)
    global_rank = np.empty(len(order), dtype="int64")
    global_rank[order] = np.arange(1, len(order) + 1)
    z_all = (combined - combined.mean()) / (combined.std() + 1e-9)
    if feats is not None:                                    # which prompt drove the match
        sims = _bld_emb @ feats.T
        z_prompt = (sims - sims.mean(axis=0)) / (sims.std(axis=0) + 1e-9)

    # Candidates in similarity order, then a soft tag filter (never empties the result)
    rel = _spatial_relation(parsed)
    cand = buildings.iloc[_bld_rows[order[:3000 if rel else max(top_k * 50, 500)]]]
    filter_note = ""
    tags = {k: v for k, v in (parsed.get("osm_tags") or {}).items()
            if not (k == "building" and v in ("yes", True))}
    if tags:
        mask, _ = _tag_match(cand, tags)
        if mask.any():
            cand = cand[mask]
        else:
            filter_note = f"OSM filter {tags} matched nothing among the visual matches and was ignored"
    targets = None
    if rel:
        cand, sp_note, targets = _apply_spatial(cand, rel)   # keeps similarity order
        filter_note = "; ".join(x for x in (filter_note, sp_note) if x)
    results = cand.head(top_k).copy()

    desc = parsed.get("visual_description") or user_query
    n, n_tiles = len(order), len(tile_rank)
    cols = {k: [] for k in ("rank", "score", "global_rank", "area_rank", "matched_on", "roads", "facts")}
    for rank, (pos, geom) in enumerate(zip(results.index, results.geometry), start=1):
        e = _row_to_emb[pos]
        g_rank, zc = int(global_rank[e]), float(z_all[e])
        strength = "very strongly" if zc >= 3.5 else "strongly" if zc >= 2.5 else "moderately"
        if feats is not None:
            matched_on = (f"your wording '{user_query}'" if z_prompt[e, 0] >= z_prompt[e, 1]
                          else f"the visual cue '{_short(desc)}'")
        else:
            matched_on = "n/a (model unavailable)"
        c = geom.centroid
        a_rank = tile_rank.get(_tile_id_at(c.x, c.y))
        near = roads.iloc[roads.sindex.query(geom.buffer(0.0015), predicate="intersects")] \
            if not roads.empty else roads
        major = near[near["highway"].isin(MAJOR_ROADS)] if "highway" in near.columns else near
        road_names = ", ".join(major["name"].dropna().astype(str).unique()[:2]) \
            if "name" in major.columns else ""
        btype = results.at[pos, "building"] if "building" in results.columns else None
        bname = results.at[pos, "name"] if "name" in results.columns else None
        what = (f"OSM {btype} building" if isinstance(btype, str) and btype != "yes"
                else "OSM building") + (f" '{bname}'" if isinstance(bname, str) else "")
        pct = g_rank / n * 100
        parts = [f"Ranked #{g_rank} of {n:,} buildings (top {pct:.2f}%)" if pct < 1 else
                 f"Ranked #{g_rank} of {n:,} buildings (top {pct:.1f}%)",
                 f"its roof imagery matches {strength}, mainly on {matched_on}", what]
        if a_rank:
            parts.append(f"surrounding 250 m area ranks #{a_rank} of {n_tiles} for this query")
        if road_names:
            parts.append(f"near {road_names}")
        if rel:
            sp = _spatial_text(results.loc[pos])
            if sp:
                parts.append(sp)
        cols["rank"].append(rank)
        cols["score"].append(round(float(combined[e]), 4))
        cols["global_rank"].append(g_rank)
        cols["area_rank"].append(a_rank)
        cols["matched_on"].append(matched_on)
        cols["roads"].append(road_names)
        cols["facts"].append("; ".join(parts) + ".")
    for k, v in cols.items():
        results[k] = v
    results["source"] = "clip_osm"
    results["filter_note"] = filter_note
    results.attrs["note"] = filter_note
    if targets is not None and len(targets):
        results.attrs["targets"] = targets
    return results

def _gap_detection(D_all, I_all, tiles, top_k):
    """Tile centres (Points) among the best visual matches that contain no OSM building."""
    buildings = _load_buildings()
    n_tiles, records = len(D_all[0]), []
    for rank, (i, score) in enumerate(zip(I_all[0][:top_k * 2], D_all[0][:top_k * 2]), start=1):
        if not 0 <= i < len(tiles):
            continue
        minx, miny, maxx, maxy = tiles[i]["bbox"]
        if len(buildings) and len(buildings.sindex.query(shapely_box(minx, miny, maxx, maxy),
                                                         predicate="intersects")):
            continue
        records.append({"geometry": Point((minx + maxx) / 2, (miny + maxy) / 2),
                        "tile_id": tiles[i]["tile_id"], "score": round(float(score), 4),
                        "area_rank": rank, "source": "gap_detection",
                        "facts": f"Area ranked #{rank} of {n_tiles} visually matches the query "
                                 "but has no building mapped in OpenStreetMap."})
    if records:
        gaps = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    else:
        gaps = gpd.GeoDataFrame(columns=["geometry", "score", "source", "facts"],
                                geometry="geometry", crs="EPSG:4326")
    gaps["rank"] = range(1, len(gaps) + 1)
    return gaps

def _load_buildings() -> gpd.GeoDataFrame:
    global _buildings
    if _buildings is None:
        try:
            b = gpd.read_file(OSM_BUILDINGS)
            b = b.set_crs("EPSG:4326") if b.crs is None else b.to_crs("EPSG:4326")
            # Keep only useful columns; the raw OSM export has ~180 sparse tag columns
            keep = [c for c in ("building", "amenity", "healthcare", "leisure", "religion", "shop",
                                "tourism", "office", "landuse", "power", "solar_panel",
                                "roof:material",
                                *NAME_COLS, "geometry") if c in b.columns]
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

def _name_match(gdf: gpd.GeoDataFrame, name_filter: str) -> gpd.GeoDataFrame:
    """Case-insensitive match of name_filter against OSM name columns.
    Pass 1: whole name as substring. Pass 2: every significant word present.
    Adds `matched_field` / `matched_name` columns for explanations."""
    cols = [c for c in NAME_COLS if c in gdf.columns]
    if gdf.empty or not cols or not name_filter.strip():
        return gdf.iloc[0:0]
    names = gdf[cols].astype("string")
    needle = name_filter.strip().lower()
    words = [w for w in needle.replace(".", " ").split() if len(w) > 2 and w not in _NAME_STOPWORDS]
    for test in (lambda col: col.str.lower().str.contains(needle, regex=False, na=False),
                 lambda col: col.str.lower().apply(
                     lambda v: bool(words) and isinstance(v, str) and all(w in v for w in words))):
        hits = names.apply(test)
        mask = hits.any(axis=1)
        if mask.any():
            out = gdf[mask].copy()
            field = hits[mask].idxmax(axis=1)  # first matching column per row
            out["matched_field"] = field.values
            out["matched_name"] = [names.loc[i, f] for i, f in zip(out.index, field)]
            return out
    return gdf.iloc[0:0]

def osm_direct_search(parsed: dict) -> gpd.GeoDataFrame:
    """Rule-based OSM search for a specifically named feature: exact OSM geometries,
    no CLIP and no tiles. Returns an empty frame if the name isn't in the data
    (never falls back to all features)."""
    name_filter = parsed.get("name_filter") or ""
    parts = []
    for layer, gdf in (("building", _load_buildings()), ("poi", _load_pois()),
                       ("road", _load_roads())):
        hit = _name_match(gdf, name_filter)
        if not hit.empty:
            hit["layer"] = layer
            parts.append(hit)
    if not parts:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    results = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry",
                               crs="EPSG:4326")
    # Tag filters only narrow the match (name is the precise selector); never empty it
    tags = {k: v for k, v in (parsed.get("osm_tags") or {}).items()
            if not (k == "building" and v in ("yes", True))}
    if tags:
        mask, _ = _tag_match(results, tags)
        if mask.any():
            results = results[mask]
    roads_hit = results[results["layer"] == "road"]
    if len(roads_hit) > 1 and "matched_name" in roads_hit.columns:
        lines = roads_hit[roads_hit.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        if len(lines):
            merged = lines.dissolve(by="matched_name", aggfunc="first", as_index=False)
            results = gpd.GeoDataFrame(pd.concat([results[results["layer"] != "road"], merged],
                                                 ignore_index=True), geometry="geometry", crs="EPSG:4326")
    results = _dedupe(results) if "name" in results.columns else results
    results["score"] = 1.0  # rule-based = full confidence
    results["source"] = "osm_direct"
    return results

def query_pipeline(user_query: str, top_k: int = TOP_K_DEFAULT, parsed: dict = None) -> gpd.GeoDataFrame:
    return search_with_tiles(user_query, top_k, parsed)[0]

def search_with_tiles(user_query: str, top_k: int = TOP_K_DEFAULT, parsed: dict = None):
    """Returns (results, tiles). results are exact OSM features (or gap points); tiles is only
    filled by the tile-level fallback used when the per-building index is missing."""
    if parsed is None:
        parsed = parse_query(user_query)

    # Route: a specifically named feature goes straight to OSM; tiles play no part
    if parsed.get("name_filter"):
        return osm_direct_search(parsed), gpd.GeoDataFrame()

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

    # Feature-level routes: tiles only confirm the appearance; geometry comes from OSM
    tile_rank = {tiles[i]["tile_id"]: r for r, i in enumerate(I_all[0], start=1)
                 if 0 <= i < len(tiles)}
    if parsed.get("task") == "gap_detection":
        return _gap_detection(D_all, I_all, tiles, top_k), gpd.GeoDataFrame()

    tags = {k: v for k, v in (parsed.get("osm_tags") or {}).items()
            if not (k == "building" and v in ("yes", True))}     # building=yes = no category
    if not tags and parsed.get("feature_type") == "road":
        tags = {"highway": "*"}
    if tags:
        tile_scores = {tiles[i]["tile_id"]: float(d) for d, i in zip(D_all[0], I_all[0])
                       if 0 <= i < len(tiles)}
        found = osm_category_search(user_query, {**parsed, "osm_tags": tags}, text_feat,
                                    tile_scores)
        if found.attrs.get("no_tag_match"):
            # The category is defined by OSM tags: if nothing carries them, say so instead
            # of returning look-alike buildings from image matching
            tag_txt = ", ".join(f"{k}={'/'.join(map(str, _as_list(v)))}" for k, v in tags.items())
            found.attrs["note"] = (f"No OpenStreetMap features tagged {tag_txt} are mapped in "
                                   "Kurla–Bandra, so there is nothing to show for this query.")
        return found, gpd.GeoDataFrame()     # may be empty after a spatial filter: say so
    if _bld_index is not None:
        return (_feature_search(user_query, parsed, text_feat, feats, top_k, tile_rank),
                gpd.GeoDataFrame())

    # Fallback (no per-building index): tile-level matches and the buildings inside them
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
        # Tiles are only a lookup index: keep the OSM footprint (left side) as the geometry
        results = results.drop(columns=["index_right"], errors="ignore").set_geometry("geometry")
        results = results.sort_values("score", ascending=False)

    # Ranked matched tiles, for the UI to highlight what the visual search actually found
    counts = results["tile_id"].value_counts() if "tile_id" in results.columns else {}
    tile_polys["n_features"] = [int(counts.get(t, 0)) for t in tile_polys["tile_id"]]
    return results, tile_polys

def get_output_formats(results: gpd.GeoDataFrame) -> dict:
    # OSM features may be Points (nodes) or Polygons/Lines (ways/relations)
    if results.empty:
        return {"geojson": "{}", "centroids": [], "bboxes": [], "count": 0}
    centroids, bboxes = [], []
    for geom in results.geometry:
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        centroids.append([round(c.y, 6), round(c.x, 6)])
        bboxes.append([round(v, 6) for v in geom.bounds])
    return {"geojson": results.to_json(), "centroids": centroids,
            "bboxes": bboxes, "count": len(results)}

__all__ = ["load_models", "query_pipeline", "search_with_tiles", "osm_direct_search",
           "osm_category_search",
           "get_output_formats"]
