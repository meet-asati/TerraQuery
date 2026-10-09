from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
import json

# Short timeout + single retry so an unreachable OpenAI endpoint falls back quickly
# instead of hanging the request (SDK default is 600s with 2 retries)
client = OpenAI(api_key=OPENAI_API_KEY or "missing", timeout=15.0, max_retries=1)

SYSTEM_PROMPT = """You are a geospatial query parser for Mumbai, India.
Convert the user's natural language query into structured JSON for searching
satellite imagery and OpenStreetMap data over the Kurla-Bandra corridor.

Return ONLY valid JSON with exactly these fields:
{
  "visual_description": "what to look for visually in satellite imagery",
  "osm_tags": {"building": "yes"},
  "spatial_relation": null,
  "feature_type": "building",
  "task": "search"
}

Rules:
- visual_description: describe satellite appearance (colours, shapes, rooftop density, texture)
- osm_tags: valid OSM keys only: building, highway, landuse, amenity, power, natural, solar_panel
- spatial_relation: null OR {"type":"near","distance_m":500,"target_osm_feature":"highway"}
- feature_type: "building" | "road" | "area" | "any"
- task: "search" | "gap_detection" | "count"
  Use gap_detection when query mentions "missing from OSM", "not on the map", "unmapped"
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
            "task": "search"
        }

__all__ = ["parse_query"]
