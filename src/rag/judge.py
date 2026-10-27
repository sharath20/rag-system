"""Ragas judge LLM backed by Ollama's native structured outputs.

Ragas' metrics ask the judge for Pydantic-shaped JSON. Small local models
(llama3.2 3B) can't follow Ragas' prompt-embedded schemas: they echo the schema
back instead of filling it. Ollama's `format` parameter constrains decoding to
the JSON schema itself, so even a 3B model always returns parseable output.
"""
import asyncio

import httpx
from pydantic import ValidationError
from ragas.llms.base import InstructorBaseRagasLLM


class OllamaJudge(InstructorBaseRagasLLM):
    def __init__(self, host: str, model: str, num_ctx: int = 8192, num_predict: int = 1024,
                 timeout: float = 600, retries: int = 1):
        self.host, self.model = host.rstrip("/"), model
        self.options = {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict}
        self.timeout, self.retries = timeout, retries

    async def agenerate(self, prompt, response_model):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": response_model.model_json_schema(),
            "stream": False,
            "options": self.options,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.retries + 1):
                resp = await client.post(f"{self.host}/api/chat", json=payload)
                resp.raise_for_status()
                try:
                    return response_model.model_validate_json(resp.json()["message"]["content"])
                except ValidationError:
                    # Truncated by num_predict or schema-valid but semantically off; retry, then give up.
                    if attempt == self.retries:
                        raise

    def generate(self, prompt, response_model):
        return asyncio.run(self.agenerate(prompt, response_model))
