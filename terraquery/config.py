import os
from dotenv import load_dotenv
load_dotenv()

# AOI — Kurla to Bandra, Mumbai
AOI_BBOX = (19.08, 19.02, 72.87, 72.82)   # north, south, east, west
AOI_CENTER = [19.05, 72.855]

# SkyCLIP
# Slim = weights only (605MB); the full training checkpoint (1.8GB, incl. optimizer state)
# can exhaust RAM on load and crash the process
SKYCLIP_CHECKPOINT = ("models/skyclip_vitb32_slim.pt"
                      if os.path.exists("models/skyclip_vitb32_slim.pt")
                      else "models/skyclip_vitb32.pt")
CLIP_MODEL = "ViT-B-32"
EMBEDDING_DIM = 512

# Tile fetching
TILE_SIZE = 256
ZOOM_LEVEL = 17
TILE_SERVER = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
TILE_SERVER_FALLBACK = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# Retrieval
TOP_K_DEFAULT = 20
SIMILARITY_THRESHOLD = 0.15

# Paths
TILE_DIR = "data/tiles"
TILE_INDEX = "data/tile_index.json"
FAISS_INDEX = "data/skyclip_tile.faiss"
OSM_BUILDINGS = "data/osm_buildings.gpkg"
OSM_ROADS = "data/osm_roads.gpkg"
OSM_POIS = "data/osm_pois.gpkg"   # amenities, healthcare, stations, parks... (often not buildings)
OSM_LANDCOVER = "data/osm_landcover.gpkg"   # water bodies, waterways, natural areas, land use
# Per-building SkyCLIP embeddings (built by build_building_index.py); rows = positions in OSM_BUILDINGS
BUILDING_FAISS = "data/skyclip_building.faiss"
BUILDING_ROWS = "data/building_rows.npy"

# OpenAI — loaded from .env, never hardcoded
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"