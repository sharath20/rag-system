"""Embedding and chat clients backed by a local Ollama server."""
import httpx


class OllamaClient:
    def __init__(self, host: str, embed_model: str, chat_model: str, timeout: float = 120):
        self.host = host.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            f"{self.host}/api/embed",
            json={"model": self.embed_model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def chat(self, prompt: str, temperature: float = 0.0) -> str:
        resp = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.chat_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
