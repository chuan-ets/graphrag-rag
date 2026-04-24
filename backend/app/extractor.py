import json
from typing import List, Tuple
from app.config import OLLAMA_HOST, OLLAMA_CHAT_MODEL, SYS_PROMPT
from openai import OpenAI


class ExtractionAgent:
    """
    Knowledge graph triple extractor using Ollama exclusively.
    """
    def __init__(self):
        self.system_prompt = SYS_PROMPT
        self.client = OpenAI(
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
        )
        self.model = OLLAMA_CHAT_MODEL

    def extract(self, text: str, existing_entities: List[str] = None) -> List[Tuple[str, str, str]]:
        content = ""
        user_msg = f"Extract triples from this text:\n\n{text}"
        if existing_entities:
            user_msg += (
                f"\n\nCURRENT ENTITIES IN GRAPH: {', '.join(existing_entities[:100])}\n"
                "IMPORTANT: If any entity you extract is synonymous with an entity in the 'CURRENT ENTITIES' list "
                "(e.g., abbreviations like 'SLS' for 'Space Launch System', or similar names), "
                "you MUST use the exact name from the list to ensure connectivity."
            )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg}
                ]
            )
            content = response.choices[0].message.content.strip()

            # Robust JSON extraction: find first '[' and last ']'
            start_idx = content.find("[")
            end_idx = content.rfind("]")

            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                content = content[start_idx:end_idx + 1]

            triples = json.loads(content)
            valid_triples = []
            if isinstance(triples, list):
                for t in triples:
                    if isinstance(t, list) and len(t) == 3:
                        s, r, o = t
                        valid_triples.append((str(s), str(r), str(o)))
            return valid_triples

        except Exception as e:
            print(f"Extraction error with Ollama model {self.model}: {e} | "
                  f"Raw content: {repr(content[:200]) if content else 'N/A'}...")
            return []


if __name__ == "__main__":
    agent = ExtractionAgent()
    print(agent.extract("MindmapGraph is a JSON-backed Knowledge Graph for incremental multi-document ingestion."))
