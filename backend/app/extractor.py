import json
from typing import List, Tuple
from deepagents import create_deep_agent
from app.config import *
import os

# Set API key for LangChain/DeepAgents
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["OLLAMA_API_BASE"] = OLLAMA_HOST

class ExtractionAgent:
    def __init__(self):
        self.system_prompt = SYS_PROMPT

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        # Define models to try in order
        models_to_try = [f"openrouter:{LLM_MAIN_MODEL}"]
        for fm in LLM_FALLBACK_MODELS:
            models_to_try.append(f"openrouter:{fm}")
        models_to_try.append(f"ollama:{OLLAMA_CHAT_MODEL}")
        
        content = ""
        for model_name in models_to_try:
            try:
                # print(f"ExtractionAgent trying model: {model_name}")
                agent = create_deep_agent(
                    model=model_name,
                    system_prompt=self.system_prompt,
                )
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": f"Extract triples from this text:\n\n{text}"}]})
                
                # The agent returns a message object. We need to extract the content.
                content = response["messages"][-1].content
                
                # More robust JSON extraction: find the first '[' and last ']'
                content = content.strip()
                start_idx = content.find("[")
                end_idx = content.rfind("]")
                
                if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                    content = content[start_idx:end_idx+1]
                
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
                print(f"Extraction error with model {model_name}: {e} | Raw content: {repr(content[:200]) if content else 'N/A'}...")
                # Continue to the next fallback model
                
        # If all models fail, return empty list
        return []

if __name__ == "__main__":
    # Test
    agent = ExtractionAgent()
    print(agent.extract("MindmapGraph is a JSON-backed Knowledge Graph for incremental multi-document ingestion."))
