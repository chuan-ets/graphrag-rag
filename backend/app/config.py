import os
from dotenv import load_dotenv

load_dotenv()

# LLM config
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_KEY") 
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://100.71.230.7:11434")
LLM_ROUTER_MODEL = os.getenv("LLM_ROUTER_MODEL", "openai/gpt-oss-120b:free")
LLM_MAIN_MODEL = os.getenv("LLM_MAIN_MODEL", "openai/gpt-oss-120b:free")

# List of fallback OpenRouter models to try before falling back to Ollama
LLM_FALLBACK_MODELS = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "google/gemini-pro-1.5-exp",
    "meta-llama/llama-3-70b-instruct:free"
]

OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen/qwen3-embedding-8b")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-documents")


# Search & Graph
TOP_K_HYBRID = 10
TOP_K_FINAL = 5
RRF_K = 60
GRAPH_FILE = "./data/knowledge_graph.json"
VECTOR_DB_DIR = "./chroma_db"
FTS_INDEX_DIR = "./fts_index"
DATA_DIR = "./data"

MIND_MAP_PATH="./data/mindmap.json"

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# Graph Agent System Prompt
SYS_PROMPT = """You are an expert knowledge graph extractor. 
Your goal is to extract meaningful entities and their relationships from the provided text.

RULES:
1. Extract ONLY significant entities (Named Entities, Technical Concepts, Systems, People, Organizations).
2. Avoid generic words like "it", "system", "process", "result" unless they are the primary subject of the document.
3. Relations should be concise (e.g., "is_a", "belongs_to", "implements", "optimized_for", "contains").
4. Output MUST be a JSON list of triples: [[subject, relation, object], ...].
5. Normalize entity names to their most common form (e.g., "Apple Inc" -> "Apple").
6. Lemmatize all entities and relations to their base form. 
This will simplify deduplication.

Example:
Input: "FastAPI is a modern web framework for Python."
Output: [["FastAPI", "is_a", "web framework"], ["FastAPI", "uses", "Python"]]
"""