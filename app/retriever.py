import os
import chromadb
#search libraries
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, ID, TEXT
from whoosh.qparser import QueryParser

from typing import List, Dict
from openai import OpenAI
from config import *
from mind_graph import MindmapGraph
from sentence_transformers import CrossEncoder

class Retriever:
    def __init__(self):
        self.chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.collection = self.chroma.get_or_create_collection(name="rag_docs")
        self._init_fts()
        self.graph = MindmapGraph(GRAPH_FILE)
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        self.embed_model_name = EMBED_MODEL
        self.rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _init_fts(self):
        os.makedirs(FTS_INDEX_DIR, exist_ok=True)
        try:
            self.fts_idx = open_dir(FTS_INDEX_DIR)
        except Exception:
            schema = Schema(id=ID(stored=True, unique=True), content=TEXT(stored=True))
            self.fts_idx = create_in(FTS_INDEX_DIR, schema)

    def openrouter_embed(self, texts):
        if isinstance(texts, str):
            res = self.llm_client.embeddings.create(model=self.embed_model_name, input=texts)
            return res.data[0].embedding
        else:
            return [self.llm_client.embeddings.create(model=self.embed_model_name, input=t).data[0].embedding for t in texts]

    def vector_search(self, query: str, k: int) -> List[Dict]:
        emb = self.openrouter_embed([query])[0]
        res = self.collection.query(query_embeddings=[emb], n_results=k, include=["metadatas", "documents", "distances"])
        return [{"id": res["ids"][0][i], "text": res["documents"][0][i], "meta": res["metadatas"][0][i], 
                 "score": 1 - res["distances"][0][i], "src": "vector"} for i in range(len(res["ids"][0]))]

    def fts_search(self, query: str, k: int) -> List[Dict]:
        with self.fts_idx.searcher() as s:
            qp = QueryParser("content", self.fts_idx.schema)
            return [{"id": r["id"], "text": r["content"], "meta": {}, "score": r.score, "src": "fts"} 
                    for r in s.search(qp.parse(query), limit=k)]

    def graph_search(self, query: str, k: int) -> List[Dict]:
        related = self.graph.expand_query_entities(query, max_hops=2)
        if not related:
            return []
        doc_ids = self.graph.get_linked_doc_ids(related)
        if not doc_ids:
            return []
        res = self.collection.get(ids=doc_ids[:k*2], include=["metadatas", "documents"])
        return [{"id": res["ids"][i], "text": res["documents"][i], "meta": res["metadatas"][i], 
                 "score": 0.75, "src": "graph"} for i in range(len(res["ids"]))]

    def rrf_fusion(self, vec: List[Dict], fts: List[Dict], graph: List[Dict]) -> List[Dict]:
        scores = {}
        def add(results, offset=0):
            for rank, item in enumerate(results):
                cid = item["id"]
                if cid not in scores:
                    scores[cid] = {**item, "rrf": 0.0, "sources": []}
                scores[cid]["rrf"] += 1.0 / (RRF_K + rank + offset)
                scores[cid]["sources"].append(item["src"])
        add(vec)
        add(fts, len(vec))
        add(graph, len(vec) + len(fts))
        return sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)[:TOP_K_HYBRID]

    def rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        pairs = [[query, d["text"]] for d in docs]
        scores = self.rerank_model.predict(pairs)
        for i, d in enumerate(docs):
            d["rerank_score"] = float(scores[i])
        return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:TOP_K_FINAL]

    def parent_join(self, docs: List[Dict]) -> List[Dict]:
        """Fetch full parent document context for retrieved chunks."""
        parent_ids = [f"{d['meta'].get('doc_id')}_full" for d in docs]
        parent_ids = list(set(parent_ids))
        if parent_ids:
            res = self.collection.get(ids=parent_ids, include=["documents", "metadatas"])
            parent_map = {m["doc_id"]: doc for doc, m in zip(res["documents"], res["metadatas"])}
            for d in docs:
                doc_id = d["meta"].get("doc_id")
                d["full_context"] = parent_map.get(doc_id, d["text"])
        return docs

    def search(self, query: str) -> List[Dict]:
        vec = self.vector_search(query, TOP_K_HYBRID)
        fts = self.fts_search(query, TOP_K_HYBRID)
        graph = self.graph_search(query, TOP_K_HYBRID)
        fused = self.rrf_fusion(vec, fts, graph)
        reranked = self.rerank(query, fused)
        return self.parent_join(reranked)[:TOP_K_FINAL]