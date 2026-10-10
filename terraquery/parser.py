from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
import json

# Short timeout + single retry so an unreachable OpenAI endpoint falls back quickly
# instead of hanging the request (SDK default is 600s with 2 retries)
client = OpenAI(api_key=OPENAI_API_KEY or "missing", timeout=15.0, max_retries=1)

SYSTEM_PROMPT = """You are a geospatial query parser for Mumbai, India.
Convert the user's natural language query into structured JSON for searching
satellite imagery and OpenStreetMap (OSM) data over the Kurla-Bandra corridor.

Return ONLY valid JSON with exactly these fields:
{
  "visual_description": "...",
  "osm_tags": {},
  "spatial_relation": null,
  "feature_type": "any",
  "task": "search",
  "name_filter": null
}

Rules:
- osm_tags: the OSM tags that identify WHAT the user asks for. Values are a string,
  a list of strings (any of them), or "*" (any value). Several keys mean "any of these".
  Use {} when the query is about APPEARANCE only (roof colour/material, density, slums,
  tin roofs, solar panels, shadows, construction sites) - those are found visually.
  NEVER default to {"building": "yes"}.
  Examples:
    hospitals -> {"amenity": "hospital", "healthcare": "hospital", "building": "hospital"}
    clinics -> {"amenity": "clinic", "healthcare": "clinic"}
    schools -> {"amenity": "school", "building": "school"}
    colleges -> {"amenity": "college", "building": "college"}
    universities -> {"amenity": "university", "building": "university"}
    temples -> {"amenity": "place_of_worship", "religion": "hindu", "building": "temple"}
    mosques -> {"amenity": "place_of_worship", "religion": "muslim", "building": "mosque"}
    churches -> {"amenity": "place_of_worship", "religion": "christian", "building": "church"}
    places of worship -> {"amenity": "place_of_worship"}
    parks / gardens -> {"leisure": ["park", "garden"]}
    playgrounds / sports grounds -> {"leisure": ["playground", "pitch", "sports_centre", "stadium"]}
    railway / train / local stations -> {"railway": ["station", "halt"],
                                         "station": ["!subway", "!monorail", "!light_rail"]}
    metro stations -> {"railway": "station", "station": "subway"}
    monorail stations -> {"railway": "station", "station": "monorail"}
    bus stops -> {"highway": "bus_stop"}
    banks / ATMs -> {"amenity": ["bank", "atm"]}
    restaurants / cafes -> {"amenity": ["restaurant", "cafe", "fast_food"]}
    petrol pumps -> {"amenity": "fuel"}
    police stations -> {"amenity": "police"}
    markets -> {"amenity": "marketplace", "shop": ["supermarket", "mall"]}
    malls -> {"shop": "mall"}
    hotels -> {"tourism": "hotel"}
    roads / streets -> {"highway": "*"}
    main / major roads -> {"highway": ["motorway", "trunk", "primary", "secondary"]}
    highways -> {"highway": ["motorway", "trunk"]}
    industrial buildings / warehouses -> {"building": ["industrial", "warehouse", "shed"]}
    apartments / residential buildings -> {"building": ["apartments", "residential", "house"]}
    dense housing / slum rooftops / blue tarpaulin roofs -> {} (visual)
- "religion" and "station" are qualifiers: they narrow the other tags, they are not
  alternatives. A value starting with "!" excludes it (e.g. "!subway").
- visual_description: how the target looks from above in satellite imagery (shape,
  colour, roof material, texture, surroundings). Always fill it in.
- feature_type: "road" for streets/roads/highways/flyovers, "building" for buildings and
  institutions, "area" for parks/grounds/water, else "any".
- spatial_relation: null, OR when the query restricts results by distance to something:
  {"type": "within" | "beyond", "distance_m": <metres>, "target": "<short label>",
   "target_osm_tags": {<OSM tags of the target, same format as osm_tags>}}
  "within 200 m of X" -> within 200; "near X" / "next to X" -> within 500;
  "more than 1 km from X" / "far from X" -> beyond (1000 / 1000). Convert km to metres.
  The main feature goes in osm_tags; the reference feature goes in target_osm_tags.
  Target examples:
    water bodies / lakes / ponds / rivers / creeks ->
      {"natural": "water", "water": "*", "waterway": ["river", "stream", "canal", "dock"]}
    railway stations -> {"railway": ["station", "halt"], "station": ["!subway", "!monorail", "!light_rail"]}
    main roads -> {"highway": ["motorway", "trunk", "primary", "secondary"]}
    parks -> {"leisure": ["park", "garden"]}
    hospitals -> {"amenity": "hospital", "healthcare": "hospital", "building": "hospital"}
  e.g. "schools within 200 m of water bodies" -> osm_tags {"amenity": "school", "building":
  "school"}, spatial_relation {"type": "within", "distance_m": 200, "target": "water bodies",
  "target_osm_tags": {"natural": "water", "water": "*", "waterway": ["river", "stream",
  "canal", "dock"]}}
- task: "search" | "gap_detection" | "count"
  Use gap_detection when query mentions "missing from OSM", "not on the map", "unmapped".
- name_filter: null OR a string. Set it ONLY when the user names one specific place,
  institution, building or road (e.g. "TSEC College", "Lilavati Hospital",
  "Western Express Highway"). Copy the name as the user wrote it, without words like
  "show me" or "where is". Use null for generic queries ("show me colleges", "slums").
  When a name is given, still fill osm_tags with the category if obvious.
"""

def parse_query(user_query: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[parser] OpenAI call failed: {e}")
        return {
            "visual_description": user_query,
            "osm_tags": {},
            "spatial_relation": None,
            "feature_type": "any",
            "task": "search",
            "name_filter": None
        }

EXPLAIN_PROMPT = """You explain why a satellite-image search selected each result (a building, or an area) in Mumbai.
For each result you get computed facts (rank, similarity strength, which cue matched,
OpenStreetMap buildings/roads). Write ONE plain sentence (max 25 words) per result saying
why it was selected for the user's query.
- Lead with what the imagery shows that fits the query, then add the most distinctive
  context for THAT result: named places, nearby roads, or notable building types.
- Do not start with "This area was selected"; vary wording between results.
- Mention building counts only when notable (e.g. none mapped, or mostly industrial).
- Use ONLY the given facts; never invent features. If the match is only "moderately",
  say the match is weaker.
Return JSON: {"reasons": [{"rank": 1, "reason": "..."}]} with one entry per result."""

def explain_tiles(user_query: str, parsed: dict, tiles) -> list:
    """One GPT sentence per matched tile; falls back to the computed facts string."""
    facts = list(tiles["facts"]) if "facts" in tiles.columns else [""] * len(tiles)
    if not len(tiles):
        return []
    payload = {
        "query": user_query,
        "visual_description": parsed.get("visual_description"),
        "areas": [{"rank": int(r), "facts": f} for r, f in zip(tiles["rank"], facts)],
    }
    try:
        response = client.with_options(timeout=10, max_retries=0).chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": EXPLAIN_PROMPT},
                {"role": "user", "content": json.dumps(payload)}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        by_rank = {int(x["rank"]): str(x["reason"]).strip()
                   for x in json.loads(response.choices[0].message.content)["reasons"]}
        return [by_rank.get(int(r)) or f for r, f in zip(tiles["rank"], facts)]
    except Exception as e:
        print(f"[parser] explain call failed, using facts: {e}")
        return facts

__all__ = ["parse_query", "explain_tiles"]
