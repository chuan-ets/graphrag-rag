import logging
logging.basicConfig(level=logging.INFO)

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tempfile
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.pipeline import RAGPipeline
from app.ingest import ingest_file
from app.config import GRAPH_FILE
from pydantic import BaseModel
from app.metrics import collector
import time
from fastapi import Request, FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="RAG Pipeline API")

class QueryRequest(BaseModel):
    query: str
    method: str = "all"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Record metrics (excluding /metrics itself to avoid noise)
    if request.url.path != "/metrics":
        collector.record_api_call(
            endpoint=request.url.path,
            method=request.method,
            duration=process_time,
            status_code=response.status_code
        )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

pipeline = RAGPipeline()

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    # Pass method to pipeline if supported, otherwise just query
    # Check if pipeline.query accepts second argument
    import inspect
    sig = inspect.signature(pipeline.query)
    if 'method' in sig.parameters:
        return pipeline.query(request.query, request.method)
    return pipeline.query(request.query)

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

@app.get("/metrics")
async def get_metrics():
    from fastapi import Response
    data, content_type = collector.get_prometheus_data()
    return Response(content=data, media_type=content_type)

@app.get("/metrics_summary")
async def get_metrics_summary():
    return collector.get_summary()

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    html_content = """
    <html>
        <head>
            <title>GraphRAG Monitoring</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: sans-serif; margin: 20px; background: #f4f4f9; }
                .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
                h1 { color: #333; }
                .stat { font-size: 24px; font-weight: bold; color: #007bff; }
                table { width: 100%; border-collapse: collapse; }
                th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>GraphRAG Monitoring Dashboard</h1>
            <div class="grid">
                <div class="card"><div>Total Requests</div><div id="total_requests" class="stat">-</div></div>
                <div class="card"><div>Avg Latency (s)</div><div id="avg_latency" class="stat">-</div></div>
                <div class="card"><div>Total Tokens</div><div id="total_tokens" class="stat">-</div></div>
                <div class="card"><div>Total Errors</div><div id="total_errors" class="stat" style="color:red">-</div></div>
            </div>
            <div class="card">
                <h2>Recent API Calls</h2>
                <table id="api_table">
                    <thead><tr><th>Time</th><th>Method</th><th>Endpoint</th><th>Status</th><th>Duration</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
            <script>
                async function updateMetrics() {
                    const res = await fetch('/metrics_summary');
                    const data = await res.json();
                    document.getElementById('total_requests').innerText = data.stats.total_requests;
                    document.getElementById('avg_latency').innerText = data.stats.avg_latency.toFixed(3);
                    document.getElementById('total_tokens').innerText = data.stats.total_tokens.toLocaleString();
                    document.getElementById('total_errors').innerText = data.stats.total_errors;
                    
                    const tbody = document.querySelector('#api_table tbody');
                    tbody.innerHTML = '';
                    data.recent_api_calls.reverse().forEach(call => {
                        const row = `<tr>
                            <td>${new Date(call.timestamp*1000).toLocaleTimeString()}</td>
                            <td>${call.method}</td>
                            <td>${call.endpoint}</td>
                            <td>${call.status_code}</td>
                            <td>${call.duration.toFixed(3)}s</td>
                        </tr>`;
                        tbody.innerHTML += row;
                    });
                }
                setInterval(updateMetrics, 2000);
                updateMetrics();
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", workers=3, port=8080)
