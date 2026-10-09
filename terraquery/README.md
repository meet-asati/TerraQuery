# TerraQuery AI — Kurla–Bandra, Mumbai

Natural language satellite imagery search powered by SkyCLIP + OpenAI.

## Setup

1. Install dependencies
   pip install -r requirements.txt

2. Ensure .env has your OpenAI key
   OPENAI_API_KEY=sk-...

3. Place SkyCLIP checkpoint at models/skyclip_vitb32.pt
   Download: curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_B32_top50pct.zip
   Then: unzip SkyCLIP_ViT_B32_top50pct.zip and move the .pt file to models/

4. Build the index (downloads tiles + OSM automatically)
   python indexer.py
   (~400-600 tiles, ~10 min)

5. Run the app
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
NL query → OpenAI gpt-4o-mini → SkyCLIP+FAISS visual search
→ OSM tag filter → gpd.sjoin spatial join → GeoJSON + Leafmap UI

## Notes
- Never commit .env
- Gap detection: left sjoin where index_right is NaN = visible but unmapped features
- All geometries EPSG:4326
- FAISS IndexFlatIP + L2-normalized embeddings = cosine similarity
