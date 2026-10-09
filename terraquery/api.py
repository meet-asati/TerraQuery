from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pipeline
from pipeline import load_models, query_pipeline, get_output_formats
from parser import parse_query

app = FastAPI(title="TerraQuery AI", version="1.0.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    load_models()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 20

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post("/search")
async def search(req: QueryRequest):
    parsed = parse_query(req.query)
    results = query_pipeline(req.query, req.top_k, parsed)
    output = get_output_formats(results)
    return {"parsed_query": parsed, **output}

@app.get("/health")
async def health():
    # Read through the module: `from pipeline import _model` would capture None at import time
    return {"status": "ok", "aoi": "Kurla-Bandra, Mumbai",
            "models_loaded": pipeline._model is not None,
            "index_loaded": pipeline._faiss_index is not None}
