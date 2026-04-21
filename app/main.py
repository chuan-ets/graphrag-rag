import logging
logging.basicConfig(level=logging.INFO)

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from app.pipeline import RAGPipeline
from app.ingest import ingest_file
from app.config import GRAPH_FILE

app = FastAPI(title="RAG Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()

@app.post("/query")
async def query_endpoint(query: str = Form(...)):
    return pipeline.query(query)

@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = ingest_file(tmp_path)
    finally:
        os.unlink(tmp_path)
    return result

@app.get("/graph")
async def get_graph():
    if os.path.exists(GRAPH_FILE):
        with open(GRAPH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"nodes": [], "links": []}