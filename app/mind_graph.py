import json
import os
import re
from typing import List, Tuple, Set
import networkx as nx
from app.config import *

class MindmapGraph:
    """JSON-backed Knowledge Graph / Mindmap for incremental multi-document ingestion."""
    def __init__(self, filepath: str = None):
        self.filepath = filepath or MIND_MAP_PATH
        self.graph = nx.DiGraph()
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
            # Convert to NetworkX graph
            self.graph = nx.node_link_graph(data)

    def save(self):
        data = nx.node_link_data(self.graph)
        for node in data.get("nodes", []):
            if "docs" in node:
                node["docs"] = list(node["docs"])
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def extract_relations(text: str) -> List[Tuple[str, str, str]]:
        """Lightweight triple extraction: (subject, relation, object)"""
        patterns = [
            (r"([\w\s\-]{2,40})\s+is\s+a?\s+([\w\s\-]{2,40})", "is_a"),
            (r"([\w\s\-]{2,40})\s+belongs to\s+([\w\s\-]{2,40})", "belongs_to"),
            (r"([\w\s\-]{2,40})\s+requires\s+([\w\s\-]{2,40})", "requires"),
            (r"([\w\s\-]{2,40})\s+of\s+([\w\s\-]{2,40})", "part_of"),
            (r"([\w\s\-]{2,40})\s+related to\s+([\w\s\-]{2,40})", "related_to")
        ]
        triples = []
        for pattern, rel in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subj, obj = match.group(1).strip().lower(), match.group(2).strip().lower()
                #if both subject and object are reasonably long, 
                # consider it a valid triple
                if len(subj) > 2 and len(obj) > 2:
                    # add to triples list (subj, relation, obj)
                    triples.append((subj, rel, obj))
        return triples

    def ingest_document(self, doc_id: str, text: str):
        """Incrementally merge document triples into the graph."""
        # Extract triples from the document text
        triples = self.extract_relations(text)
        # For each triple, add nodes and edges to the graph.
        for subj, rel, obj in triples:
            for node in [subj, obj]:
                if not self.graph.has_node(node):
                    self.graph.add_node(node, type="entity", mentions=0, docs=set())
                self.graph.nodes[node]["mentions"] += 1
                self.graph.nodes[node]["docs"].add(doc_id)

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