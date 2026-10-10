import html
import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
# folium backend: the default ipyleaflet widget renders blank inside Streamlit
import leafmap.foliumap as leafmap
from config import AOI_CENTER
from pipeline import load_models, search_with_tiles
from parser import parse_query, explain_tiles

def feature_label(f):
    """Best human label for one result row."""
    if f.get("source") == "gap_detection":
        return "Unmapped area"
    name, btype, tag = f.get("name"), f.get("building"), f.get("matched_tag")
    if isinstance(name, str):
        return name
    if isinstance(f.get("group_label"), str):   # merged unnamed pieces, e.g. "Unnamed rail (yard)"
        return f["group_label"]
    if isinstance(tag, str) and "=" in tag and not tag.startswith("building="):
        return "Unnamed " + tag.split("=", 1)[1].replace("_", " ")
    if isinstance(btype, str) and btype != "yes":
        return f"{btype.capitalize()} building"
    return "Building"

st.set_page_config(layout="wide", page_title="TerraQuery AI")

# Load model + data once per server process (no-op on later reruns)
with st.spinner("Loading model and map data (first start only)..."):
    load_models()

st.markdown("""<style>
header[data-testid="stHeader"] { display: none; }
.block-container { padding: 1rem 1.5rem 0.5rem; }
.stTextInput input { border-radius: 8px; }
.tq-list { max-height: 330px; overflow-y: auto; padding-right: 4px; }
.tq-item { padding: 6px 0; border-bottom: 1px solid #e3e6ee; font-size: 0.9rem; }
.tq-item code { font-size: 0.8rem; }
.tq-summary { font-size: 0.95rem; margin: 2px 0 6px; }
.tq-why { margin-top: 3px; line-height: 1.35; }
.tq-facts { margin-top: 2px; font-size: 0.78rem; color: #6b7280; line-height: 1.3; }
</style>""", unsafe_allow_html=True)

# Session state
if "results_gdf" not in st.session_state:
    st.session_state.results_gdf = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []

col_map, col_panel = st.columns([2, 1], gap="medium")

# Map is filled after the search runs, so results show in the same run (no st.rerun loop)
map_slot = col_map.container()

with col_panel:
    panel = st.container(border=True)

with panel:
    st.markdown("#### 🛰 TerraQuery AI")
    st.caption("Kurla–Bandra, Mumbai · Natural language satellite search")

    # Search bar (Enter or → submits)
    with st.form("search", border=False):
        q_col, btn_col = st.columns([5, 1])
        typed = q_col.text_input("Search query", placeholder="e.g. railway tracks and railway yard",
                                 label_visibility="collapsed")
        submitted = btn_col.form_submit_button("→", use_container_width=True)

    # Example query buttons
    ex1, ex2, ex3 = st.columns(3)
    query = typed.strip() if submitted and typed.strip() else ""
    if ex1.button("☀ Solar panels", use_container_width=True):
        query = "houses with solar panels on rooftop"
    if ex2.button("🔍 Missing OSM", use_container_width=True):
        query = "buildings visible in imagery but missing from OpenStreetMap"
    if ex3.button("🏘 Dense housing", use_container_width=True):
        query = "dense residential housing"

    # Optional deep link: ?q=<query> runs once on first load
    url_q = st.experimental_get_query_params().get("q", [""])[0]
    if not query and url_q and not st.session_state.get("url_q_done"):
        query = url_q
        st.session_state.url_q_done = True

    if query:
        with st.spinner("Searching satellite imagery..."):
            parsed = parse_query(query)
            results_gdf, tiles_gdf = search_with_tiles(query, top_k=10, parsed=parsed)
            if not tiles_gdf.empty:
                tiles_gdf["reason"] = explain_tiles(query, parsed, tiles_gdf)
            elif (not results_gdf.empty and "facts" in results_gdf.columns
                  and "reason" not in results_gdf.columns):  # tag results carry exact reasons
                results_gdf["reason"] = explain_tiles(query, parsed, results_gdf)
            st.session_state.search_note = results_gdf.attrs.pop("note", "") or ""
            st.session_state.targets_gdf = results_gdf.attrs.pop("targets", None)
            st.session_state.results_gdf, st.session_state.tiles_gdf = results_gdf, tiles_gdf
            st.session_state.last_parsed = parsed
            st.session_state.query_history.append(query)

    results = st.session_state.results_gdf
    tiles = st.session_state.get("tiles_gdf")
    parsed = st.session_state.get("last_parsed") or {}
    named = bool(parsed.get("name_filter"))
    source = (results["source"].iloc[0] if results is not None and len(results)
              and "source" in results.columns else None)
    gap = parsed.get("task") == "gap_detection"
    feature_mode = not named and (source in ("clip_osm", "gap_detection", "osm_tags") or
                                  (gap and results is not None and not len(tiles if tiles is not None else [])))

    if results is None:
        st.info("Type a query above or pick an example")
    elif named:
        # Named feature: exact OSM match (no imagery tiles involved)
        name_q = parsed["name_filter"]
        if results.empty:
            st.info(f"“{name_q}” was not found in the OpenStreetMap data for Kurla–Bandra. "
                    "It may be outside the indexed area or mapped under another name.")
        else:
            st.markdown(
                f"<div class='tq-summary'><b>“{html.escape(name_q)}”</b><br>"
                f"<b>{len(results)}</b> exact OSM match{'es' if len(results) != 1 else ''} · "
                f"source: <code>OpenStreetMap (name search)</code></div>",
                unsafe_allow_html=True)
            rows = []
            for _, f in results.head(20).iterrows():
                c = f.geometry.centroid
                kind = f.get("building") if f.get("layer") == "building" else f.get("highway")
                kind = kind if isinstance(kind, str) else f.get("layer", "feature")
                why = (f"Matched by name: OSM field <code>{html.escape(str(f['matched_field']))}</code> "
                       f"= “{html.escape(str(f['matched_name']))}” ({html.escape(str(kind))}, "
                       f"{f.geometry.geom_type.lower()})")
                rows.append(f"<div class='tq-item'><b>{html.escape(str(f['matched_name']))}</b> · "
                            f"<code>{c.y:.5f}, {c.x:.5f}</code><div class='tq-why'>💡 {why}</div></div>")
            st.markdown(f"<div class='tq-list'>{''.join(rows)}</div>", unsafe_allow_html=True)
            cols = [c for c in ("matched_name", "matched_field", "layer", "building", "amenity",
                                "highway", "source", "score", "geometry") if c in results.columns]
            export = results[cols].copy()
            export.insert(0, "query", st.session_state.query_history[-1])
            export["center_lat"] = [round(g.centroid.y, 6) for g in export.geometry]
            export["center_lon"] = [round(g.centroid.x, 6) for g in export.geometry]
            st.download_button("⬇ Download GeoJSON", export.to_json(), "results.geojson",
                               "application/json", use_container_width=True)
    elif feature_mode:
        # Individually ranked OSM features (per-building embeddings) or gap points
        q = st.session_state.query_history[-1]
        if results.empty:
            st.info("No unmapped areas among the top visual matches — every matching area already "
                    "has buildings in OpenStreetMap." if gap else
                    (st.session_state.get("search_note") or
                     "No matching features found — try another query"))
        else:
            if source == "osm_tags":
                count = int(results["total_matches"].iloc[0])
                tags_txt = ", ".join(sorted(set(results["matched_tag"].dropna().astype(str)))[:4])
                what = f"OSM features found ({tags_txt})" + (
                    f" · showing {len(results)}" if count > len(results) else "")
            else:
                what = "unmapped areas (no OSM building)" if gap else \
                    "individual buildings (best of all mapped buildings)"
                count = len(results)
            note = st.session_state.get("search_note") or (
                results["filter_note"].iloc[0] if "filter_note" in results.columns else "")
            st.markdown(
                f"<div class='tq-summary'><b>“{html.escape(q)}”</b><br>"
                f"<b>{count}</b> {html.escape(what)}"
                + (f"<br><span class='tq-facts'>{html.escape(note)}</span>" if note else "")
                + "</div>", unsafe_allow_html=True)
            rows = []
            for _, f in results.iterrows():
                c = f.geometry.representative_point()
                score_txt = (f"score <code>{float(f['score']):.3f}</code> · "
                             if pd.notna(f.get("score")) else "")
                rows.append(
                    f"<div class='tq-item'><b>#{f['rank']} {html.escape(feature_label(f))}</b> · "
                    f"{score_txt}<code>{c.y:.5f}, {c.x:.5f}</code>"
                    f"<div class='tq-why'>💡 {html.escape(str(f.get('reason', f.get('facts', ''))))}</div>"
                    f"<div class='tq-facts'>{html.escape(str(f.get('facts', '')))}</div></div>")
            st.markdown(f"<div class='tq-list'>{''.join(rows)}</div>", unsafe_allow_html=True)
            cols = [c for c in ("rank", "score", "name", "matched_tag", "layer", "building",
                                "amenity", "highway", "source", "reason", "facts", "global_rank",
                                "area_rank", "roads", "total_matches", "geometry")
                    if c in results.columns]
            export = results[cols].copy()
            export.insert(0, "query", q)
            export.insert(2, "label", [feature_label(f) for _, f in results.iterrows()])
            export["center_lat"] = [round(g.representative_point().y, 6) for g in export.geometry]
            export["center_lon"] = [round(g.representative_point().x, 6) for g in export.geometry]
            st.download_button("⬇ Download GeoJSON", export.to_json(), "results.geojson",
                               "application/json", use_container_width=True)
    elif tiles is None or tiles.empty:
        st.info(st.session_state.get("search_note") or
                "No matching features found — try another query")
    else:
        st.markdown(
            f"<div class='tq-summary'><b>“{st.session_state.query_history[-1]}”</b><br>"
            f"<b>{len(tiles)}</b> matched tiles · <b>{len(results)}</b> "
            f"{'unmapped tiles' if gap else 'buildings'} · filter: "
            f"<code>{parsed.get('osm_tags') or 'none'}</code></div>",
            unsafe_allow_html=True)
        items = "".join(
            f"<div class='tq-item'><b>#{t['rank']}</b> · score <code>{t['score']:.3f}</code> · "
            f"{t['n_features']} features · <code>{t.geometry.centroid.y:.5f}, "
            f"{t.geometry.centroid.x:.5f}</code>"
            f"<div class='tq-why'>💡 {html.escape(str(t.get('reason', '')))}</div>"
            f"<div class='tq-facts'>{html.escape(str(t.get('facts', '')))}</div></div>"
            for _, t in tiles.iterrows())
        st.markdown(f"<div class='tq-list'>{items}</div>", unsafe_allow_html=True)
        # Export only the matched locations (ranked tiles) with readable details
        export = tiles[["rank", "score", "n_features", "geometry"]].copy()
        export.insert(0, "query", st.session_state.query_history[-1])
        export["score"] = export["score"].round(4)
        bounds = export.geometry.bounds  # tiles are rectangles: bbox centre == centroid
        export["center_lat"] = ((bounds.miny + bounds.maxy) / 2).round(6)
        export["center_lon"] = ((bounds.minx + bounds.maxx) / 2).round(6)
        export = export.rename(columns={"n_features": "buildings_in_area"})
        for col in ("reason", "facts", "building_mix", "named_places", "roads"):
            if col in tiles.columns:
                export[col] = tiles[col].values
        st.download_button("⬇ Download GeoJSON", export.to_json(), "results.geojson",
                           "application/json", use_container_width=True)

with map_slot:
    m = leafmap.Map(center=AOI_CENTER, zoom=14,
                    draw_control=False, measure_control=False, fullscreen_control=False)
    m.add_basemap("Esri.WorldImagery")
    if named and results is not None and not results.empty:
        # Exact OSM footprint(s) for a named feature; zoom to them
        tip = [c for c in ("matched_name", "layer", "building", "highway", "geometry")
               if c in results.columns]
        m.add_gdf(results[tip], layer_name="Exact match",
                  style={"color":"#e63946","fillColor":"#e63946",
                         "weight":3,"fillOpacity":0.45})
        b = results.total_bounds
        pad = 0.0015  # ~150 m so the footprint is shown in context
        m.fit_bounds([[b[1] - pad, b[0] - pad], [b[3] + pad, b[2] + pad]])
    targets = st.session_state.get("targets_gdf")
    if feature_mode and targets is not None and len(targets) and results is not None and len(results):
        tcols = [c for c in ("target_name", "target_tag", "geometry") if c in targets.columns]
        t_poly = targets[targets.geometry.geom_type != "Point"]
        if len(t_poly):
            m.add_gdf(t_poly[tcols], layer_name="Reference features",
                      style={"color": "#1d70b8", "fillColor": "#4ea8de",
                             "weight": 3, "fillOpacity": 0.35})
        for _, t in targets[targets.geometry.geom_type == "Point"].iterrows():
            folium.CircleMarker([t.geometry.y, t.geometry.x], radius=7, color="#1d70b8",
                                fill=True, fill_opacity=0.8,
                                tooltip=str(t.get("target_name") or t.get("target_tag"))).add_to(m)
    if feature_mode and results is not None and not results.empty:
        # One numbered pin per matched feature, plus its exact OSM footprint
        polys = results[results.geometry.geom_type != "Point"]
        if len(polys):
            m.add_gdf(polys[["rank", "score", "geometry"]], layer_name="Matched buildings",
                      style={"color":"#e63946","fillColor":"#e63946",
                             "weight":3,"fillOpacity":0.45})
        for _, f in results.iterrows():
            c = f.geometry.representative_point()
            folium.Marker(
                [c.y, c.x],
                tooltip=f"#{f['rank']} {feature_label(f)} — {f.get('reason', f.get('facts', ''))}",
                icon=folium.DivIcon(icon_size=(24, 24), icon_anchor=(12, 12), html=(
                    "<div style='background:#e63946;color:#fff;border:2px solid #fff;"
                    "border-radius:50%;width:24px;height:24px;line-height:20px;text-align:center;"
                    f"font:600 12px sans-serif;box-shadow:0 1px 4px #0006'>{f['rank']}</div>"))
            ).add_to(m)
        b = results.total_bounds
        pad = 0.0015
        m.fit_bounds([[b[1] - pad, b[0] - pad], [b[3] + pad, b[2] + pad]])
    elif not named and results is not None and not results.empty and "building" in results.columns:
        keep = [c for c in ("building", "name", "score", "geometry") if c in results.columns]
        m.add_gdf(results[keep].head(500), layer_name="Buildings",
                  style={"color":"#e63946","fillColor":"#e63946",
                         "weight":1,"fillOpacity":0.35})
    if tiles is not None and not tiles.empty:
        # Matched tiles: what the visual search found, ranked by similarity
        tip_cols = [c for c in ("rank", "score", "n_features", "reason", "geometry") if c in tiles.columns]
        m.add_gdf(tiles[tip_cols], layer_name="Matched tiles",
                  style={"color":"#ffd60a","fillColor":"#ffd60a",
                         "weight":3,"fillOpacity":0.08})
        b = tiles.total_bounds
        m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    # Render in memory: leafmap's to_streamlit() round-trips through a temp file read with the
    # Windows default encoding, which crashes on non-Latin OSM names (e.g. Devanagari)
    m.add_layer_control()
    components.html(m.get_root().render(), height=640)
