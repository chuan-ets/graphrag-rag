import logging
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import OLLAMA_HOST, OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL
from app.metrics import collector


class FallbackLLM:
    """
    LLM wrapper using Ollama exclusively via its OpenAI-compatible API.
    """
    def __init__(self):
        self.client = OpenAI(
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",  # Required by the SDK but ignored by Ollama
        )

    def chat_completion(self, primary_model: str, messages: List[Dict], **kwargs) -> Any:
        """
        Perform a chat completion. primary_model is accepted for API compatibility
        but Ollama always uses OLLAMA_CHAT_MODEL unless overridden via env.
        """
        model = OLLAMA_CHAT_MODEL
        logging.info(f"Chat completion with Ollama model: {model}")
        start = time.time()
        try:
            res = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            duration = time.time() - start
            collector.record_llm_call(model, "chat", duration, True)
            
            # Record token usage if available
            if hasattr(res, 'usage') and res.usage:
                collector.record_llm_tokens(
                    res.usage.prompt_tokens, 
                    res.usage.completion_tokens, 
                    model
                )
            return res
        except Exception as e:
            collector.record_llm_call(model, "chat", time.time() - start, False)
            raise e

    def embed(self, primary_model: str, input_texts: Any) -> Any:
        """
        Generate embeddings using Ollama. primary_model is accepted for API
        compatibility but Ollama always uses OLLAMA_EMBED_MODEL.
        """
        model = OLLAMA_EMBED_MODEL
        logging.info(f"Embedding with Ollama model: {model}")
        start = time.time()
        try:
            res = self.client.embeddings.create(
                model=model,
                input=input_texts
            )
            collector.record_llm_call(model, "embed", time.time() - start, True)
            return res
        except Exception as e:
            collector.record_llm_call(model, "embed", time.time() - start, False)
            raise e
