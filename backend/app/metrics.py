import time
from collections import deque
from typing import Dict, List, Any
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

class MetricsCollector:
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.api_calls = deque(maxlen=history_size)
        self.llm_calls = deque(maxlen=history_size)
        self.errors = deque(maxlen=history_size)
        self.start_time = time.time()
        
        # Prometheus Metrics
        self.p_api_requests = Counter('api_requests_total', 'Total API Requests', ['method', 'endpoint', 'status'])
        self.p_api_latency = Histogram('api_request_duration_seconds', 'API Request Latency', ['endpoint'])
        self.p_llm_calls = Counter('llm_calls_total', 'Total LLM Calls', ['model', 'type', 'success'])
        self.p_llm_latency = Histogram('llm_call_duration_seconds', 'LLM Call Latency', ['model', 'type'])
        self.p_errors = Counter('system_errors_total', 'Total System Errors', ['context'])
        self.p_tokens = Counter('llm_tokens_total', 'Total LLM Tokens Used', ['model', 'type']) # type: prompt or completion

        # Aggregated stats (for internal dashboard)
        self.stats = {
            "total_requests": 0,
            "total_llm_calls": 0,
            "total_errors": 0,
            "avg_latency": 0.0,
            "total_tokens": 0,
            "last_updated": time.time()
        }

    def record_api_call(self, endpoint: str, method: str, duration: float, status_code: int):
        call_data = {
            "timestamp": time.time(),
            "endpoint": endpoint,
            "method": method,
            "duration": duration,
            "status_code": status_code
        }
        self.api_calls.append(call_data)
        self.stats["total_requests"] += 1
        
        # Update Prometheus
        self.p_api_requests.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
        self.p_api_latency.labels(endpoint=endpoint).observe(duration)
        
        # Update running average latency
        n = self.stats["total_requests"]
        curr_avg = self.stats["avg_latency"]
        self.stats["avg_latency"] = (curr_avg * (n - 1) + duration) / n
        self.stats["last_updated"] = time.time()

    def record_llm_tokens(self, prompt_tokens: int, completion_tokens: int, model: str):
        self.p_tokens.labels(model=model, type='prompt').inc(prompt_tokens)
        self.p_tokens.labels(model=model, type='completion').inc(completion_tokens)
        self.stats["total_tokens"] += (prompt_tokens + completion_tokens)
        self.stats["total_llm_calls"] += 1

    def record_llm_call(self, model: str, call_type: str, duration: float, success: bool):
        call_data = {
            "timestamp": time.time(),
            "model": model,
            "type": call_type, 
            "duration": duration,
            "success": success
        }
        self.llm_calls.append(call_data)
        self.stats["total_llm_calls"] += 1
        if not success:
            self.stats["total_errors"] += 1
        self.stats["last_updated"] = time.time()

        # Update Prometheus
        self.p_llm_calls.labels(model=model, type=call_type, success=str(success)).inc()
        self.p_llm_latency.labels(model=model, type=call_type).observe(duration)

    def record_error(self, error_msg: str, context: str = ""):
        self.errors.append({
            "timestamp": time.time(),
            "message": error_msg,
            "context": context
        })
        self.stats["total_errors"] += 1
        self.p_errors.labels(context=context).inc()

    def get_prometheus_data(self):
        return generate_latest(), CONTENT_TYPE_LATEST

    def get_summary(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": int(uptime),
            "stats": self.stats,
            "recent_api_calls": list(self.api_calls)[-10:], 
            "recent_llm_calls": list(self.llm_calls)[-10:],
            "recent_errors": list(self.errors)[-5:]
        }

# Global collector instance
collector = MetricsCollector()
