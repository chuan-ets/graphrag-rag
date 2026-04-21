from langchain_community.document_loaders import SitemapLoader
from app.mind_graph import MindmapGraph
from urllib.parse import urlparse
import asyncio

class SitemapGraphBuilder:
    def __init__(self, graph: MindmapGraph):
        self.graph = graph

    async def ingest_sitemap(self, sitemap_url: str):
        """Fetch sitemap and build parent-child relations for URLs."""
        loader = SitemapLoader(web_path=sitemap_url)
        # We only need the metadata/URLs for the structure
        docs = loader.load()
        
        for doc in docs:
            url = doc.metadata.get("source")
            if not url:
                continue
            
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split("/") if p]
            
            # Create a hierarchy of nodes
            # Example: /docs/api/auth -> [docs, api, auth]
            # Relations: docs -> parent_of -> api, api -> parent_of -> auth
            
            prev_node = parsed.netloc # Start with domain
            if not self.graph.graph.has_node(prev_node):
                self.graph.graph.add_node(prev_node, type="domain", docs=set())

            current_path = ""
            for part in path_parts:
                current_path += f"/{part}"
                node_name = f"{parsed.netloc}{current_path}"
                
                if not self.graph.graph.has_node(node_name):
                    self.graph.graph.add_node(node_name, type="web_page", docs=set(), label=part)
                
                # Add edge
                if not self.graph.graph.has_edge(prev_node, node_name, key="HAS_SUBPAGE"):
                    self.graph.graph.add_edge(prev_node, node_name, key="HAS_SUBPAGE", relation="HAS_SUBPAGE")
                
                prev_node = node_name
        
        self.graph.save()
        return len(docs)

if __name__ == "__main__":
    # Test with a small sitemap if needed
    # builder = SitemapGraphBuilder(MindmapGraph())
    # asyncio.run(builder.ingest_sitemap("https://python.langchain.com/sitemap.xml"))
    pass
