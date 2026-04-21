import json
from typing import List, Tuple
from deepagents import create_deep_agent
from app.config import OPENROUTER_API_KEY, LLM_MAIN_MODEL, SYS_PROMPT
import os

# Set API key for LangChain/DeepAgents
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

class ExtractionAgent:
    def __init__(self):
        system_prompt = SYS_PROMPT
        self.agent = create_deep_agent(
            model=f"openrouter:{LLM_MAIN_MODEL}",
            system_prompt=system_prompt,
        )

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        try:
            response = self.agent.invoke(
                # {"messages": [{"role": "user", "content": f"Extract triples from this text:\n\n{text}"}]})
                {"messages": [{"role": "user", "content": f"Extract triples from this text:\n\n{text}"}]})
            # The agent returns a message object. We need to extract the content.
            content = response["messages"][-1].content
            
            # Simple cleaning of the response to ensure it's valid JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            triples = json.loads(content)
            # Ensure it's a list of triples
            valid_triples = []
            if isinstance(triples, list):
                for t in triples:
                    if isinstance(t, list) and len(t) == 3:
                        # Ensure all 3 elements are strings
                        s, r, o = t
                        valid_triples.append((str(s), str(r), str(o)))
            return valid_triples
        except Exception as e:
            print(f"Extraction error: {e}")
            return []

if __name__ == "__main__":
    # Test
    agent = ExtractionAgent()
    print(agent.extract("MindmapGraph is a JSON-backed Knowledge Graph for incremental multi-document ingestion."))
