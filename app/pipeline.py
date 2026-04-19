import ollama
from typing import Dict, List
import os
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.config import *
from pydantic import BaseModel
from config import *
from retriever import Retriever
from mind_graph import MindmapGraph

class RouterOutput(BaseModel):
    intent: str
    domain: str
    use_graph: bool
    confidence: float

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.ollama = ollama.Client(host=settings.ollama_host)
        self.graph = MindmapGraph(settings.graph_file)

    def route(self, query: str) -> RouterOutput:
        prompt = f"""Classify the query. Return JSON only.
Query: "{query}"
Fields: intent (query|unknown), domain (general|technical|financial), use_graph (bool), confidence (0.0-1.0)"""
        res = self.ollama.chat(model=settings.llm_router_model, messages=[{"role": "user", "content": prompt}], format="json")
        return RouterOutput(**res["message"]["content"])

    def build_context(self, query: str, docs: List[Dict]) -> str:
        parts = []
        mm_hint = self.graph.get_mindmap_context(query)
        if mm_hint:
            parts.append(mm_hint)
        for i, d in enumerate(docs, 1):
            parts.append(f"[{i}] ({', '.join(d['sources'])}) {d['full_context'][:800]}...")
        return "\n\n".join(parts)

    def generate(self, query: str, context: str) -> str:
        prompt = f"""You are a precise AI assistant. Answer ONLY using the provided context.
If information is missing, state: "Not found in provided documents."
Always cite sources using [1], [2] format.

CONTEXT:
{context}

QUESTION: {query}
ANSWER:"""
        res = self.ollama.chat(model=settings.llm_main_model, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.1})
        return res["message"]["content"]

    def query(self, user_query: str) -> Dict:
        router = self.route(user_query)
        if router.intent == "unknown" or router.confidence < 0.3:
            return {"answer": "Query unclear. Please rephrase.", "router": router.model_dump(), "sources": []}
        
        docs = self.retriever.search(user_query)
        if not docs:
            return {"answer": "No relevant documents found.", "router": router.model_dump(), "sources": []}
        
        context = self.build_context(user_query, docs)
        answer = self.generate(user_query, context)
        
        return {
            "answer": answer,
            "router": router.model_dump(),
            "sources": [{"id": d["id"], "score": d.get("rerank_score", 0), "sources": d["sources"]} for d in docs]
        }