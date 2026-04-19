import uuid
import os
from minio import Minio
from sentence_transformers import SentenceTransformer
from whoosh import index, fields
from whoosh.writing import AsyncWriter
from config import *
from mind_graph import MindmapGraph
from retriever import Retriever

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append({"text": " ".join(words[i:i + chunk_size]), "start": i, "end": i + chunk_size})
    return chunks

def ingest_file(file_path: str) -> dict:
    doc_id = str(uuid.uuid4())
    filename = os.path.basename(file_path)
    
    # 1. MinIO Storage
    client = Minio(settings.minio_endpoint, access_key=settings.minio_access_key,
                   secret_key=settings.minio_secret_key, secure=False)
    if not client.bucket_exists(settings.minio_bucket):
        client.make_bucket(settings.minio_bucket)
    client.fput_object(settings.minio_bucket, f"{doc_id}/{filename}", file_path)
    
    # 2. Chunk & Embed
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    embed_model = SentenceTransformer("BAAI/bge-m3")
    embs = embed_model.encode([c["text"] for c in chunks]).tolist()
    
    # 3. ChromaDB (Store full doc + chunks)
    retriever = Retriever()
    retriever.collection.add(
        embeddings=[embed_model.encode([text])[0].tolist()],
        documents=[text],
        metadatas=[{"doc_id": doc_id, "filename": filename, "is_full": True}],
        ids=[f"{doc_id}_full"]
    )
    meta_list = [{"doc_id": doc_id, "filename": filename, "chunk_idx": i, **c} for i, c in enumerate(chunks)]
    retriever.collection.add(embeddings=embs, documents=[c["text"] for c in chunks], 
                             metadatas=meta_list, ids=[f"{doc_id}_c{i}" for i in range(len(chunks))])
    
    # 4. FTS Index
    idx = index.open_dir(settings.fts_index_dir) if True else index.create_in(settings.fts_index_dir, fields.TEXT(stored=True))
    writer = AsyncWriter(idx)
    for i, c in enumerate(chunks):
        writer.add_document(id=f"{doc_id}_c{i}", content=c["text"])
    writer.commit()
    
    # 5. Mindmap Graph Update
    mindmap = MindmapGraph(settings.graph_file)
    for c in chunks:
        mindmap.ingest_document(doc_id, c["text"])
    
    return {"status": "success", "doc_id": doc_id, "chunks": len(chunks)}