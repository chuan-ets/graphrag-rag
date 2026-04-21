from openai import OpenAI
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
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        self.graph = MindmapGraph(GRAPH_FILE)

    def route(self, query: str) -> RouterOutput:
        prompt = f"""Classify the query. Return JSON only.
Query: "{query}"
Fields: intent (query|unknown), domain (general|technical|financial), use_graph (bool), confidence (0.0-1.0)"""
        res = self.llm_client.chat.completions.create(
            model=LLM_ROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        try:
            data = json.loads(res.choices[0].message.content)
            return RouterOutput(**data)
        except Exception:
            return RouterOutput(intent="unknown", domain="general", use_graph=False, confidence=0.0)

    def build_context(self, query: str, docs: List[Dict]) -> str:
        parts = []
        mm_hint = self.graph.get_mindmap_context(query)
        if mm_hint:
            parts.append(mm_hint)
        for i, d in enumerate(docs, 1):
            meta = d.get("meta", {})
            filename = meta.get("filename", "Unknown")
            chunk = meta.get("chunk_idx", "?")
            parts.append(f"[{i}] (File: {filename}, Page/Section: {chunk}) {d['text']}")
        return "\n\n".join(parts)

    def generate(self, query: str, context: str) -> str:
        prompt = f"""You are a precise AI assistant. Answer ONLY using the provided context.
If information is missing, state: "Not found in provided documents."
When answering, ALWAYS cite your sources using the marker (e.g. [1]) and mention the File name and Page/Section to be helpful.

CONTEXT:
{context}

QUESTION: {query}
ANSWER:"""
        res = self.llm_client.chat.completions.create(
            model=LLM_MAIN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return res.choices[0].message.content

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