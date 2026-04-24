from openai import OpenAI
from typing import Dict, List
import os
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.config import *
from app.llm_wrapper import FallbackLLM
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
        self.llm_client = FallbackLLM()
        self.graph = MindmapGraph(GRAPH_FILE)

    def route(self, query: str) -> RouterOutput:
        prompt = f"""Classify the query. Return JSON only.
Query: "{query}"
Fields: intent (query|unknown), domain (general|technical|financial), use_graph (bool), confidence (0.0-1.0)"""
        res = self.llm_client.chat_completion(
            primary_model=LLM_ROUTER_MODEL,
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
        for d in docs:
            meta = d.get("meta", {})
            filename = meta.get("filename", "Unknown")
            parts.append(f"SOURCE: {filename}\nCONTENT: {d['text']}")
        return "\n\n".join(parts)

    def generate(self, query: str, context: str) -> str:
        prompt = f"""You are a precise AI assistant. Answer ONLY using the provided context.
If information is missing, state: "Not found in provided documents."
When answering, ALWAYS cite your sources using the exact filename in square brackets, e.g. [filename.txt].

CONTEXT:
{context}

QUESTION: {query}
ANSWER:"""
        res = self.llm_client.chat_completion(
            primary_model=LLM_MAIN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return res.choices[0].message.content

    def query(self, user_query: str, method: str = "all") -> Dict:
        router = self.route(user_query)
        if router.intent == "unknown" or router.confidence < 0.3:
            return {"answer": "Query unclear. Please rephrase.", "router": router.model_dump(), "sources": []}
        
        docs = self.retriever.search(user_query, method=method)
        if not docs:
            return {"answer": "No relevant documents found.", "router": router.model_dump(), "sources": []}
        
        context = self.build_context(user_query, docs)
        answer = self.generate(user_query, context)
        
        return {
            "answer": answer,
            "router": router.model_dump(),
            "sources": [{"id": d["id"], "score": d.get("rerank_score", 0), "filename": d.get("meta", {}).get("filename", "Unknown"), "sources": d["sources"]} for d in docs]
        }