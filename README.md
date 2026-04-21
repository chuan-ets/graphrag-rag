# GraphRAG Pipeline API

This is an advanced Retrieval-Augmented Generation (RAG) system that combines Knowledge Graph traversal, Vector Search, and Full-Text Search (FTS) to build highly contextual answers. It exposes a set of REST APIs using FastAPI and utilizes cloud LLMs and embeddings via **OpenRouter** (using the `openai` Python SDK).

## Features

- **Hybrid Search + Graph Search**: Combines ChromaDB vector search, Whoosh full-text search, and NetworkX-based mindmap/knowledge graph search using Reciprocal Rank Fusion (RRF).
- **Knowledge Graph Generation**: Extracts triples (subject-relation-object) automatically from documents to incrementally build a mindmap.
- **Query Routing**: Employs an LLM router to classify the intent and domain of the user query before retrieval.
- **Parent Document Retrieval (Parent Join)**: Fetches the entire document context after finding the most relevant chunks to maintain contextual integrity.
- **Cloud AI Models**: Uses OpenRouter to access powerful generation and embedding models seamlessly without local hardware constraints.
- **Document Storage**: Uses MinIO to store ingested raw files.

## Project Structure

- `app/main.py`: FastAPI application entry point, containing `/ingest` and `/query` endpoints.
- `app/pipeline.py`: Defines the main RAG pipeline (`RAGPipeline`), responsible for query routing, context building, and generating final answers via OpenRouter.
- `app/retriever.py`: Manages the complex retrieval logic combining Vector, FTS, and Graph searches (`Retriever`). Implements RRF fusion and Parent Join.
- `app/ingest.py`: Handles document parsing (PDFs, TXT), chunking, storing raw files in MinIO, generating embeddings via OpenRouter, creating Whoosh FTS indexes, and extracting triples for the Knowledge Graph.
- `app/mind_graph.py`: Manages the NetworkX Knowledge Graph, loads/saves the graph to a JSON file, and provides BFS graph expansion algorithms to find related entities.
- `app/config.py`: Environment and configuration variables (OpenRouter API keys, MinIO credentials, Chroma DB host, model names).
- `docker-compose.yml`: Contains services for ChromaDB and MinIO.

## Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose** (for MinIO and ChromaDB)
- **OpenRouter API Key**: Sign up at [OpenRouter](https://openrouter.ai/) to get your API key.

## Setup & Installation

1. **Start the Infrastructure**
   Start MinIO and ChromaDB using Docker Compose:
   ```bash
   docker-compose up -d
   ```

2. **Setup Python Environment**
   Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the root directory (or update the existing one) with the following parameters:
   ```env
   # OpenRouter
   OPEN_ROUTER_KEY=your-openrouter-api-key-here
   LLM_ROUTER_MODEL=openai/gpt-oss-120b:free
   LLM_MAIN_MODEL=openai/gpt-oss-120b:free
   EMBED_MODEL=qwen/qwen3-embedding-8b
   
   # MinIO
   MINIO_ENDPOINT=localhost:9000
   MINIO_ACCESS_KEY=minioadmin
   MINIO_SECRET_KEY=minioadmin
   MINIO_BUCKET=rag-documents
   
   # ChromaDB
   CHROMA_HOST=localhost
   CHROMA_PORT=8000
   ```

## Usage

### Start the API Server
Run the FastAPI application via uvicorn:
```bash
uvicorn app.main:app --port 8080
```

### Ingest Documents
You can ingest `.txt` or `.pdf` files. The system will store the file in MinIO, chunk it, embed it via OpenRouter, and build the FTS and Knowledge Graph indexes.
```bash
curl -X POST -F "file=@/path/to/your/document.pdf" http://localhost:8080/ingest
```

### Query the System
Ask questions against your ingested documents:
```bash
curl -X POST -F "query=What is O-RAN?" http://localhost:8080/query
```

## How It Works (Retrieval Process)

1. **Routing**: The user's query is classified by a lightweight LLM router.
2. **Retrieval**: 
   - **Vector Search**: Embeds the query and queries ChromaDB.
   - **FTS Search**: Performs a keyword search via Whoosh.
   - **Graph Search**: Expands the query's entities by 2 hops in the Mindmap Graph to find semantically related documents.
3. **Fusion**: Results from the three methods are merged using Reciprocal Rank Fusion (RRF).
4. **Parent Join**: The system fetches the original parent chunks of the retrieved vectors for full context.
5. **Generation**: The context is fed into the main LLM to generate a precise, cited answer.
