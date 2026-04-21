from langchain_community.document_loaders import PyPDFLoader
import uuid
import os
from minio import Minio
from openai import OpenAI
from whoosh import index, fields
from whoosh.writing import AsyncWriter
from config import *
from mind_graph import MindmapGraph
from retriever import Retriever
from extractor import ExtractionAgent

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append({"text": " ".join(words[i:i + chunk_size]), "start": i, "end": i + chunk_size})
    return chunks

def ingest_file(file_path: str) -> dict:
    doc_id = str(uuid.uuid4())
    filename = os.path.basename(file_path)
    
    # MinIO Storage
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                   secret_key=MINIO_SECRET_KEY, secure=False)
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
    client.fput_object(MINIO_BUCKET, f"{doc_id}/{filename}", file_path)
    
    # Chunk & Embed
    if file_path.lower().endswith('.pdf'):
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # Gộp text các trang PDF lại, mỗi trang là một chunk
            chunks = [{"text": doc.page_content, "start": 0, "end": 0} for doc in docs]
            text = "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return {"status": "error", "message": f"Could not read PDF: {e}"}
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            return {"status": "error", "message": f"Could not read file as UTF-8 text: {e}"}
        chunks = chunk_text(text)
    llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    embed_model_name = EMBED_MODEL
    embs = [llm_client.embeddings.create(model=embed_model_name, input=c["text"]).data[0].embedding for c in chunks]
    
    # ChromaDB (Store full doc + chunks)
    retriever = Retriever()
    # Embed full doc with Ollama
    full_emb = llm_client.embeddings.create(model=embed_model_name, input=text).data[0].embedding
    retriever.collection.add(
        embeddings=[full_emb],
        documents=[text],
        metadatas=[{"doc_id": doc_id, "filename": filename, "is_full": True}],
        ids=[f"{doc_id}_full"]
    )
    meta_list = [{"doc_id": doc_id, "filename": filename, "chunk_idx": i, **c} for i, c in enumerate(chunks)]
    retriever.collection.add(embeddings=embs, documents=[c["text"] for c in chunks], 
                             metadatas=meta_list, ids=[f"{doc_id}_c{i}" for i in range(len(chunks))])
    
    # FTS Index
    idx = index.open_dir(FTS_INDEX_DIR) if True else index.create_in(FTS_INDEX_DIR, fields.TEXT(stored=True))
    writer = AsyncWriter(idx)
    for i, c in enumerate(chunks):
        writer.add_document(id=f"{doc_id}_c{i}", content=c["text"])
    writer.commit()
    
    # Mindmap Graph Update (Agentic Extraction)
    mindmap = MindmapGraph(GRAPH_FILE)
    extractor = ExtractionAgent()
    for c in chunks:
        triples = extractor.extract(c["text"])
        if triples:
            mindmap.add_triples(doc_id, triples, source_name=filename)
    
    return {"status": "success", "doc_id": doc_id, "chunks": len(chunks)}