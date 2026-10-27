"""Embedding and chat clients backed by a local Ollama server."""
import httpx


class OllamaClient:
    def __init__(self, host: str, embed_model: str, chat_model: str, timeout: float = 120,
                 num_ctx: int = 8192, num_predict: int = 1024):
        self.host = host.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.timeout = timeout
        # Ollama's default context (4096) silently truncates 5 x 650-token passages plus the
        # repair prompt; num_predict stops small models from looping into endless output.
        self.num_ctx = num_ctx
        self.num_predict = num_predict

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
                "options": {"temperature": temperature, "num_ctx": self.num_ctx, "num_predict": self.num_predict},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def client_from_config(cfg: dict) -> OllamaClient:
    o = cfg["ollama"]
    return OllamaClient(o["host"], o["embed_model"], o["chat_model"], o.get("timeout", 120),
                        o.get("num_ctx", 8192), o.get("num_predict", 1024))
