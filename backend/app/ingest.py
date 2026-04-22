from langchain_community.document_loaders import PyPDFLoader
import uuid
import os
from minio import Minio
from app.llm_wrapper import FallbackLLM
from whoosh import index, fields
from whoosh.writing import AsyncWriter
from config import *
from mind_graph import MindmapGraph
from retriever import Retriever
from extractor import ExtractionAgent

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    #split the text in chunks of chunk_size with overlap of overlap
    #make a list of words "I am student" -> ["i","am","student"]
    #join the words in chunks of chunk_size with overlap of overlap
    
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
    #check bucket exist
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
    client.fput_object(MINIO_BUCKET, f"{doc_id}/{filename}", file_path)
    
    # Chunk & Embed
    if file_path.lower().endswith('.pdf'):
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            chunks = []
            for i, page in enumerate(pages):                
                page_text = page.page_content
                if len(page_text.split()) > CHUNK_SIZE:
                    sub_chunks = chunk_text(page_text)
                    for sc in sub_chunks:
                        sc["page_idx"] = i
                        chunks.append(sc)
                else:
                    chunks.append({"text": page_text, "start": 0, "end": 0, "page_idx": i})
            text = "\n".join([p.page_content for p in pages])
        except Exception as e:
            return {"status": "error", "message": f"Could not read PDF: {e}"}
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            return {"status": "error", "message": f"Could not read file as UTF-8 text: {e}"}
        chunks = chunk_text(text)

    llm_client = FallbackLLM()
    embed_model_name = EMBED_MODEL
    # Batch Embedding for better performance
    texts_to_embed = [c["text"] for c in chunks]
    embs = []
    batch_size = 20 
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i + batch_size]
        res = llm_client.embed(primary_model=embed_model_name, input_texts=batch)
        embs.extend([e.embedding for e in res.data])
    
    # ChromaDB (Store full doc + chunks)
    retriever = Retriever()
    # Embed full doc
    full_emb = llm_client.embed(primary_model=embed_model_name, input_texts=text).data[0].embedding
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