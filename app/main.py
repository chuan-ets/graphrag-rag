import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from app.pipeline import RAGPipeline
from app.ingest import ingest_file

app = FastAPI(title="RAG Pipeline API")
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