import os
from dotenv import load_dotenv

load_dotenv()

# LLM config
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_KEY") 
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://100.71.230.7:11434")
LLM_ROUTER_MODEL = os.getenv("LLM_ROUTER_MODEL", "openai/gpt-oss-120b:free")
LLM_MAIN_MODEL = os.getenv("LLM_MAIN_MODEL", "openai/gpt-oss-120b:free")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen/qwen3-embedding-8b")
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