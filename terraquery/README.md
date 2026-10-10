# TerraQuery AI — Kurla–Bandra, Mumbai

Natural language satellite imagery search powered by SkyCLIP + OpenAI.

## Setup

1. Install dependencies
   pip install -r requirements.txt
   For an NVIDIA GPU (used by step 5), install the CUDA build of torch afterwards:
   pip install --no-deps torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121

2. Ensure .env has your OpenAI key
   OPENAI_API_KEY=sk-...

3. Place SkyCLIP checkpoint at models/skyclip_vitb32.pt
   Download: curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_B32_top50pct.zip
   Then: unzip SkyCLIP_ViT_B32_top50pct.zip and move the .pt file to models/

4. Build the index (downloads tiles + OSM buildings, roads and POIs automatically)
   python indexer.py
   (~400-600 tiles, ~10 min)

5. Build the per-building index (SkyCLIP crop around every OSM building)
   python build_building_index.py
   (~13k buildings, ~1-2 min on GPU; re-run whenever step 4 is re-run)

6. Run the app
   streamlit run app.py

   Or as API:
   uvicorn api:app --reload --port 8000

## Example Queries
- "slum rooftops near railway lines"
- "buildings visible in imagery but missing from OpenStreetMap"
- "solar panels on residential rooftops"
- "industrial sheds near Kurla station"
- "dense housing with no road access"

## Architecture
NL query → OpenAI gpt-4o-mini (parse) →
- named place ("TSEC College") → OSM name lookup → exact footprint
- category ("hospitals", "temples", "main roads") → OSM tags across buildings, POIs
  and roads → every tagged feature with its name (roads merged per name)
- appearance query ("tin roofs", "slums") → SkyCLIP text embedding vs per-building crop embeddings (FAISS)
  → top individual OSM buildings (tile ranking only confirms the area)
- gap detection → best-matching tiles with no OSM building → tile-centre points
→ GPT "why selected" sentence per result → GeoJSON + Leafmap UI

## Notes
- Never commit .env
- Geometry always comes from OSM (or a tile-centre point for gaps), never a tile box
- All geometries EPSG:4326
- FAISS IndexFlatIP + L2-normalized embeddings = cosine similarity
