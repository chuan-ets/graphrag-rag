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
    # Check if file already exists
    try:
        existing = pipeline.retriever.collection.get(where={"filename": file.filename, "is_full": True})
        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            return {"status": "error", "message": f"File '{file.filename}' has already been uploaded."}
    except Exception as e:
        logging.warning(f"Failed to check for existing file: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = ingest_file(tmp_path)
    finally:
        os.unlink(tmp_path)
    return result

@app.get("/files")
async def list_files():
    try:
        results = pipeline.retriever.collection.get(where={"is_full": True}, include=["metadatas"])
        files = []
        for meta in results.get("metadatas", []):
            if meta:
                files.append({
                    "doc_id": meta.get("doc_id"),
                    "filename": meta.get("filename")
                })
        # Deduplicate files by doc_id
        unique_files = list({f["doc_id"]: f for f in files}.values())
        return {"status": "success", "files": unique_files}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/graph")
async def get_graph():
    if os.path.exists(GRAPH_FILE):
        with open(GRAPH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"nodes": [], "links": []}