from app.extractor import ExtractionAgent
from app.mind_graph import MindmapGraph
import os

def test_extraction():
    test_text = """
    GraphRAG utilizes LangChain Deep Agents to perform iterative retrieval. 
    It stores knowledge in a MindmapGraph, which is a JSON-backed network.
    The system is designed to handle complex queries by decomposing them into sub-tasks.
    """
    
    print("--- Starting Agentic Extraction Test ---")
    agent = ExtractionAgent()
    triples = agent.extract(test_text)
    
    print(f"Extracted {len(triples)} triples:")
    for t in triples:
        print(f"  - {t}")
        
    if triples:
        # Test graph injection
        test_graph_path = "./data/test_mindmap.json"
        if os.path.exists(test_graph_path):
            os.remove(test_graph_path)
            
        graph = MindmapGraph(test_graph_path)
        graph.add_triples("test_doc_1", triples, source_name="test_source.txt")
        print(f"\nSaved to {test_graph_path}")
        print(f"Nodes in graph: {len(graph.graph.nodes)}")
        print(f"Edges in graph: {len(graph.graph.edges)}")

if __name__ == "__main__":
    test_extraction()
