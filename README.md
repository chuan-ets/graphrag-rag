# GraphRAG Pipeline & Explorer

This project is a state-of-the-art **Retrieval-Augmented Generation (RAG)** application. It goes beyond simple vector search by combining **Vector Embeddings**, **Full-Text Search (FTS)**, and a dynamically generated **Knowledge Graph** to provide highly contextual and accurate answers. 

It comes with a fully-featured **ChatGPT-style Web UI** that allows you to chat with your documents, upload new files, and visualize the underlying Knowledge Graph in an interactive popup modal.

---

## 🌟 Key Features

### 🧠 Advanced RAG Backend
- **Hybrid Retrieval Strategy**: Merges results from ChromaDB (Vector Search), Whoosh (Keyword/FTS Search), and NetworkX (Graph Traversal) using **Reciprocal Rank Fusion (RRF)**.
- **Agentic Knowledge Extraction**: Utilizes `deepagents` and LangChain to autonomously extract entities and relationships (triples) from ingested documents to build a Knowledge Graph.
- **Semantic Query Routing**: An LLM router evaluates the user's query intent and domain before deciding how to retrieve information.
- **Cross-Encoder Reranking**: Uses `ms-marco-MiniLM-L-6-v2` to rerank the fused results for maximum relevance.
- **Parent-Child Document Retrieval**: Retrieves small, dense chunks for search, but injects the larger parent document context into the LLM prompt to preserve meaning.
- **Cloud LLM Integration**: Fully integrated with **OpenRouter** for both embedding generation and conversational completion, removing the need for heavy local GPUs.

### 💻 Interactive Frontend
- **ChatGPT-Style Interface**: A clean, centered chat interface for querying your documents.
- **Knowledge Graph Explorer Modal**: Click "View Knowledge Graph" to open an interactive, physics-based visualization of your mindmap (powered by Vis.js).
- **Live Document Management**: Drag-and-drop file ingestion (PDF, TXT) with a real-time list of uploaded files directly in the UI.

---

## 📁 Architecture & File Structure

### Backend (`/backend/app`)
- **`main.py`**: The FastAPI application entry point. Exposes endpoints for `/ingest`, `/query`, `/graph`, and `/files`.
- **`pipeline.py`**: The orchestrator (`RAGPipeline`). Handles query routing, context building, and interacting with the LLM to generate the final answer.
- **`retriever.py`**: The core search engine (`Retriever`). Executes Vector, FTS, and Graph searches, fuses them with RRF, and applies Cross-Encoder reranking.
- **`ingest.py`**: The data pipeline. Handles document parsing, chunking, MinIO storage, embedding generation, Whoosh indexing, and triggers the `ExtractionAgent`.
- **`extractor.py`**: A `deepagents`-powered LLM agent that reads text chunks and outputs JSON arrays of Subject-Relation-Object triples.
- **`mind_graph.py`**: Manages the NetworkX Knowledge Graph, saving/loading state, and providing Breadth-First Search (BFS) expansion for queries.
- **`config.py`**: Centralized configuration management via `.env` variables.

### Frontend (`/frontend`)
- **`templates/index.html`**: The main layout, structured with a centered chat area and a hidden modal for the graph.
- **`static/style.css`**: Styling rules for the ChatGPT-like UI, animations, and the glassmorphism design system.
- **`static/main.js`**: Handles API communication, chat history rendering, file uploads, file list fetching, and the Vis.js network graph initialization.

---

## 🛠️ Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose** (for MinIO and ChromaDB)
- **OpenRouter API Key**: Sign up at [OpenRouter](https://openrouter.ai/) to get your API key.

---

## 🚀 Setup & Installation

### 1. Start the Infrastructure
Start the MinIO (Object Storage) and ChromaDB (Vector Database) services using Docker Compose from the root directory:
```bash
docker compose up -d
```

### 2. Setup Python Environments
We recommend separate environments for the backend and frontend if they have different dependencies, but you can also use one. 

**Backend Setup:**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the `backend/` directory with the following configuration:
```env
# OpenRouter Configuration
OPEN_ROUTER_KEY=your-openrouter-api-key-here
LLM_ROUTER_MODEL=openai/gpt-oss-120b:free
LLM_MAIN_MODEL=openai/gpt-oss-120b:free
EMBED_MODEL=qwen/qwen3-embedding-8b

# MinIO Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=rag-documents

# ChromaDB Vector Store
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

---

## 🎮 Running the Application

### 1. Start the Backend API
In the `backend` directory, activate your virtual environment and run the FastAPI server:
```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8080
```
*The API will be available at http://localhost:8080*

### 2. Start the Frontend
In the `frontend` directory, serve the application.
```bash
source .frontend/bin/activate
python3 -m app
```

---

## 📚 API Endpoints Overview

- `POST /ingest`: Accepts a `multipart/form-data` file upload. Checks for duplicates, saves to MinIO, chunks the text, embeds it into ChromaDB, indexes it in Whoosh, and extracts graph triples.
- `POST /query`: Accepts a `query` string. Routes the query, searches databases, fuses results, and returns the LLM-generated answer along with citations.
- `GET /files`: Returns a deduplicated JSON list of all documents successfully ingested into the system.
- `GET /graph`: Returns the current state of the Knowledge Graph (nodes and edges) formatted for Vis.js frontend rendering.

---

## 💡 How the Retrieval Process Works

1. **Routing**: The user's query is classified by a lightweight LLM router to determine intent and domain.
2. **Hybrid Retrieval**: 
   - **Vector Search**: Embeds the query and performs a cosine similarity search in ChromaDB.
   - **FTS Search**: Performs a fast keyword/BM25 search via Whoosh.
   - **Graph Search**: Identifies entities in the query and expands them by 2 hops in the Mindmap Graph to find semantically related chunks.
3. **Fusion (RRF)**: Results from the three distinct retrieval methods are mathematically merged using Reciprocal Rank Fusion to balance exact matches with semantic relevance.
4. **Reranking**: A Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) re-scores the top fused results against the original query to ensure maximum relevance.
5. **Parent Join**: The system looks up the original, larger parent chunks of the retrieved snippets to provide the LLM with complete, unbroken context.
6. **Generation**: The compiled context is fed into the main LLM to generate a precise, cited answer.
