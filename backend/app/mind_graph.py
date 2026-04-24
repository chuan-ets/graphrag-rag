import json
import os
import re
from typing import List, Tuple, Set
import networkx as nx
from app.config import *
from app.extractor import ExtractionAgent

class MindmapGraph:
    """JSON-backed Knowledge Graph / Mindmap for incremental multi-document ingestion."""
    def __init__(self, filepath: str = None):
        self.filepath = filepath or MIND_MAP_PATH
        self.graph = nx.MultiDiGraph()
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self._load()

    def _load(self):
        #load graph from json file, if exists. Convert doc lists to sets for easier merging
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Restore sets from lists
            for node in data.get("nodes", []):
                if "docs" in node:
                    node["docs"] = set(node["docs"])
            for edge in data.get("links", []) + data.get("edges", []):
                if "docs" in edge:
                    edge["docs"] = set(edge["docs"])
            # Convert to NetworkX graph
            self.graph = nx.node_link_graph(data)

    def save(self):
        data = nx.node_link_data(self.graph)
        for node in data.get("nodes", []):
            if "docs" in node:
                node["docs"] = list(node["docs"])
        for edge in data.get("links", []) + data.get("edges", []):
            if "docs" in edge:
                edge["docs"] = list(edge["docs"])
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def normalize_name(self, name) -> str:
        """Standardize entity names for consistency and merge similar ones."""
        if isinstance(name, list):
            name = " ".join([str(i) for i in name])
        if not isinstance(name, str):
            name = str(name)
            
        name = name.lower().strip()
        # Remove generic prefixes
        prefixes = ["the ", "a ", "an "]
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p):]
        
        # --- Smart Fuzzy Merging ---
        # If this name is very similar to an existing entity, use the existing one
        existing_entities = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"]
        for existing in existing_entities:
            # Case 1: One is a substring of another (e.g. "sls" and "sls rocket")
            if len(existing) >= 3 and len(name) >= 3:
                if existing in name or name in existing:
                    return existing
            # Case 2: Very high overlap (simple heuristic)
            if existing.replace(" ", "") == name.replace(" ", ""):
                return existing
                
        return name


    def add_triples(self, doc_id: str, triples: List[Tuple[str, str, str]], source_name: str = None):
        """Add pre-extracted high-quality triples to the graph."""
        doc_node = source_name if source_name else doc_id
        if not self.graph.has_node(doc_node):
            self.graph.add_node(doc_node, type="document", mentions=1, docs={doc_id})

        for subj, rel, obj in triples:
            subj = self.normalize_name(subj)
            obj = self.normalize_name(obj)
            rel = rel.strip().upper().replace(" ", "_")

            if not subj or not obj:
                continue

            for node in [subj, obj]:
                if not self.graph.has_node(node):
                    self.graph.add_node(node, type="entity", mentions=0, docs=set())
                self.graph.nodes[node]["mentions"] += 1
                self.graph.nodes[node]["docs"].add(doc_id)
                
                # Link Entity to Document Node
                if not self.graph.has_edge(node, doc_node, key="MENTIONED_IN"):
                    self.graph.add_edge(node, doc_node, key="MENTIONED_IN", weight=1, docs={doc_id}, relation="MENTIONED_IN")
                else:
                    self.graph[node][doc_node]["MENTIONED_IN"]["weight"] += 1
                    self.graph[node][doc_node]["MENTIONED_IN"]["docs"].add(doc_id)

            if not self.graph.has_edge(subj, obj, key=rel):
                self.graph.add_edge(subj, obj, key=rel, weight=1, docs={doc_id}, relation=rel)
            else:
                self.graph[subj][obj][rel]["weight"] += 1
                self.graph[subj][obj][rel]["docs"].add(doc_id)
        self.save()

    def ingest_document(self, doc_id: str, text: str, source_name: str = None):
        """Incrementally merge document triples into the graph."""
        doc_node = source_name if source_name else doc_id
        if not self.graph.has_node(doc_node):
            self.graph.add_node(doc_node, type="document", mentions=1, docs={doc_id})
            
        # Get list of existing entities to help LLM map/merge them
        existing_entities = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"]
        
        # Extract triples from the document text, providing context of existing graph
        extractor = ExtractionAgent()
        triples = extractor.extract(text, existing_entities=existing_entities)
        
        # For each triple, add nodes and edges to the graph.
        for subj, rel, obj in triples:
            subj = self.normalize_name(subj)
            obj = self.normalize_name(obj)
            rel = rel.strip().upper().replace(" ", "_")
            
            if not subj or not obj:
                continue

            for node in [subj, obj]:
                if not self.graph.has_node(node):
                    self.graph.add_node(node, type="entity", mentions=0, docs=set())
                self.graph.nodes[node]["mentions"] += 1
                self.graph.nodes[node]["docs"].add(doc_id)
                
                # Link Entity to Document Node
                if self.graph.has_edge(node, doc_node, key="MENTIONED_IN"):
                    self.graph[node][doc_node]["MENTIONED_IN"]["weight"] += 1
                    self.graph[node][doc_node]["MENTIONED_IN"]["docs"].add(doc_id)
                else:
                    self.graph.add_edge(node, doc_node, key="MENTIONED_IN", weight=1, docs={doc_id}, relation="MENTIONED_IN")

            if self.graph.has_edge(subj, obj, key=rel):
                self.graph[subj][obj][rel]["weight"] += 1
                self.graph[subj][obj][rel]["docs"].add(doc_id)
            else:
                self.graph.add_edge(subj, obj, key=rel, weight=1, docs={doc_id}, relation=rel)
        self.save()

    def expand_query_entities(self, query: str, max_hops: int = 2) -> List[str]:
        """BFS traversal to find semantically related entities."""
        seeds = [w.lower() for w in re.findall(r"\b[\w\-]{3,}\b", query) if self.graph.has_node(w)]
        if not seeds:
            return []
        related = set()
        for seed in seeds:
            lengths = nx.single_source_shortest_path_length(self.graph, seed, cutoff=max_hops)
            for node, dist in lengths.items():
                if 0 < dist <= max_hops:
                    related.add(node)
        return list(related)

    def get_linked_doc_ids(self, entities: List[str]) -> List[str]:
        """Return doc_ids that mention any of the given entities."""
        doc_ids = set()
        for ent in entities:
            if self.graph.has_node(ent):
                doc_ids.update(self.graph.nodes[ent].get("docs", set()))
        return list(doc_ids)

    def get_mindmap_context(self, query: str) -> str:
        """Generate a compact, LLM-ready mindmap summary."""
        related = self.expand_query_entities(query, max_hops=2)
        if not related:
            return ""
        lines = [" MINDMAP CONTEXT:"]
        for ent in related[:6]:
            neighbors = list(self.graph.neighbors(ent))
            if neighbors:
                rels = ", ".join(neighbors[:3])
                lines.append(f"- {ent} → linked to: {rels}")
        return "\n".join(lines)