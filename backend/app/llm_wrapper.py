import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import (
    OPENROUTER_API_KEY, 
    OPENROUTER_BASE_URL, 
    OLLAMA_HOST, 
    LLM_FALLBACK_MODELS, 
    OLLAMA_CHAT_MODEL, 
    OLLAMA_EMBED_MODEL
)

class FallbackLLM:
    """
    A wrapper around OpenAI clients that provides an automatic fallback mechanism.
    Tries the primary OpenRouter model, then fallback OpenRouter models, and finally Ollama.
    """
    def __init__(self):
        self.or_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        self.ollama_client = OpenAI(
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama", # Required by the SDK but ignored by Ollama
        )

    def chat_completion(self, primary_model: str, messages: List[Dict], **kwargs) -> Any:
        # Build the list of models to try
        models_to_try = [("openrouter", primary_model)]
        
        for fm in LLM_FALLBACK_MODELS:
            models_to_try.append(("openrouter", fm))
            
        models_to_try.append(("ollama", OLLAMA_CHAT_MODEL))
        
        last_exception = None
        
        for provider, model_name in models_to_try:
            try:
                logging.info(f"Attempting chat completion with {provider} model: {model_name}")
                if provider == "openrouter":
                    return self.or_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        **kwargs
                    )
                elif provider == "ollama":
                    return self.ollama_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        **kwargs
                    )
            except Exception as e:
                logging.warning(f"{provider} model {model_name} failed: {e}")
                last_exception = e
                
        # If all models failed, raise the last exception
        raise last_exception

    def embed(self, primary_model: str, input_texts: Any) -> Any:
        # For embeddings, we just try OpenRouter, then Ollama
        models_to_try = [
            ("openrouter", primary_model),
            ("ollama", OLLAMA_EMBED_MODEL)
        ]
        
        last_exception = None
        
        for provider, model_name in models_to_try:
            try:
                logging.info(f"Attempting embedding with {provider} model: {model_name}")
                if provider == "openrouter":
                    return self.or_client.embeddings.create(
                        model=model_name,
                        input=input_texts
                    )
                elif provider == "ollama":
                    return self.ollama_client.embeddings.create(
                        model=model_name,
                        input=input_texts
                    )
            except Exception as e:
                logging.warning(f"Embedding with {provider} model {model_name} failed: {e}")
                last_exception = e
                
        raise last_exception
