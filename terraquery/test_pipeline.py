from dotenv import load_dotenv
load_dotenv()
from pipeline import load_models, query_pipeline, get_output_formats
from parser import parse_query

load_models()

def run(q):
    parsed = parse_query(q)
    print(f"\n=== {q}")
    print("name_filter:", parsed.get("name_filter"))
    results = query_pipeline(q, parsed=parsed)
    print(f"Results: {len(results)} features")
    if len(results):
        print("Geometry types:", results.geometry.geom_type.value_counts().to_dict())
        print("Source:", results["source"].iloc[0] if "source" in results.columns else "clip_pipeline")
        print("Geometry columns:", [c for c in results.columns if str(results[c].dtype) == "geometry"])
        if "matched_name" in results.columns:
            print("Matched:", results[["matched_name", "matched_field", "layer"]].head(3).to_dict("records"))
    out = get_output_formats(results)
    print("Output count:", out["count"], "| first centroid:", out["centroids"][:1])
    return parsed, results

# 1. Named feature that exists in the Kurla-Bandra OSM data -> exact OSM footprint, no tiles
p1, r1 = run("TSEC College")
assert p1.get("name_filter") and len(r1) and (r1["source"] == "osm_direct").all()

# 2. Named feature outside the indexed area -> empty result, never "all buildings"
p2, r2 = run("Lokmanya Tilak College of Engineering")
assert p2.get("name_filter") and len(r2) == 0

# 3. Appearance query -> individually ranked OSM buildings (per-building embeddings), not a cluster
from config import TOP_K_DEFAULT
p3, r3 = run("slum with dense small tin roofs")
assert not p3.get("name_filter") and 0 < len(r3) <= TOP_K_DEFAULT
assert (r3["source"] == "clip_osm").all() and (r3["score"] > 0).all()
assert r3["score"].nunique() > 1, "every feature has its own score (not a copied tile score)"
assert "index_right" not in r3.columns and "facts" in r3.columns

# 4. Gap detection -> tile-centre Points only (may be empty: this AOI is densely mapped)
p4, r4 = run("buildings missing from OpenStreetMap")
assert p4.get("task") == "gap_detection"
assert len(r4) == 0 or ((r4["source"] == "gap_detection").all()
                        and set(r4.geometry.geom_type) == {"Point"})
# 5. Category query -> OSM features carrying the tag, with real names (not random roofs)
p5, r5 = run("show me hospitals")
assert len(r5) and (r5["source"] == "osm_tags").all()
assert r5["matched_tag"].str.contains("hospital").all()
assert r5["name"].astype(str).str.contains("Hospital", case=False).any()

# 6. Road query -> road lines from the roads layer, merged per named road
p6, r6 = run("main roads")
assert len(r6) and (r6["layer"] == "road").all()
assert set(r6.geometry.geom_type) <= {"LineString", "MultiLineString"}

# 7. Named POI that is not a building (was missing before the POI layer)
p7, r7 = run("Lilavati Hospital")
assert len(r7) and r7["matched_name"].str.contains("Lilavati").any()
# 8. Railway stations = train stations only (metro/monorail stops share railway=station)
p8, r8 = run("railway stations")
assert len(r8) and not r8["name"].isin(["Bandra Kurla Complex", "Bandra Colony", "Dharavi",
                                         "Shitala Devi Mandir", "Antop Hill",
                                         "Acharya Atre Nagar"]).any()
assert r8["name"].isin(["Bandra", "Khar Road", "Sion"]).any()

# 9. Spatial relation: only schools within 200 m of a water body, with the distance
p9, r9 = run("schools within 200 m of water bodies")
assert p9.get("spatial_relation") and len(r9)
assert (r9["dist_m"] <= 200).all() and r9["matched_tag"].str.contains("school").all()
assert int(r9["total_matches"].iloc[0]) < 149   # fewer than all schools
print("\nAll checks passed.")
