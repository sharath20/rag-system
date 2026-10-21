"""Cross-encoder re-ranking, run through ONNX Runtime (no torch needed).

The model scores each (question, passage) pair jointly, which is much more
precise than comparing two independently computed embeddings. It is too slow to
run over the whole corpus, so it only re-orders the hybrid retriever's candidates.
"""
from .store import Hit


class NoopReranker:
    """Keeps the incoming (fused) order. Used when re-ranking is disabled."""

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        return hits[:top_k]


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 512,
                 onnx_file: str = "onnx/model.onnx"):
        self.model, self.max_length, self.onnx_file = model, max_length, onnx_file
        self._session = None
        self._tok = None

    def _load(self):
        if self._session is not None:
            return
        # Lazy imports: only pay for these when re-ranking is actually used.
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(hf_hub_download(self.model, "tokenizer.json"))
        self._tok.enable_truncation(max_length=self.max_length)
        self._tok.enable_padding()
        self._session = ort.InferenceSession(
            hf_hub_download(self.model, self.onnx_file), providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        import numpy as np

        self._load()
        enc = self._tok.encode_batch([(query, p) for p in passages])
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
        }
        logits = self._session.run(None, {k: v for k, v in feed.items() if k in self._inputs})[0]
        return [float(x) for x in logits.reshape(len(passages), -1)[:, 0]]

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        for h, s in zip(hits, self.score(query, [h.text for h in hits])):
            h.scores["rerank"] = s
        return sorted(hits, key=lambda h: h.scores["rerank"], reverse=True)[:top_k]


def build_reranker(cfg: dict):
    r = cfg.get("reranker", {})
    if not r.get("enabled", False):
        return NoopReranker()
    return CrossEncoderReranker(r.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"), r.get("max_length", 512))
