import html
import streamlit as st
# folium backend: the default ipyleaflet widget renders blank inside Streamlit
import leafmap.foliumap as leafmap
from config import AOI_CENTER
from pipeline import load_models, search_with_tiles
from parser import parse_query, explain_tiles

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
            st.session_state.results_gdf, st.session_state.tiles_gdf = results_gdf, tiles_gdf
            st.session_state.last_parsed = parsed
            st.session_state.query_history.append(query)

    results = st.session_state.results_gdf
    tiles = st.session_state.get("tiles_gdf")

    if results is None:
        st.info("Type a query above or pick an example")
    elif tiles is None or tiles.empty:
        st.info("No matching features found — try another query")
    else:
        parsed = st.session_state.get("last_parsed") or {}
        gap = parsed.get("task") == "gap_detection"
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
    if results is not None and not results.empty and "building" in results.columns:
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
    m.to_streamlit(height=640)
