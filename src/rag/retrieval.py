"""Hybrid retrieval: BM25 + vector search, fused with Reciprocal Rank Fusion, then re-ranked.

  question --> vector top-N --\
                               +--> RRF fuse --> cross-encoder re-rank --> top_k
  question --> BM25 top-N ----/

BM25 catches exact identifiers (CORSMiddleware, BackgroundTasks, status codes)
that embeddings blur; vectors catch paraphrases. RRF combines the two by rank
only, so their incompatible score scales never need calibrating.
"""
from .bm25 import BM25
from .rerank import NoopReranker
from .store import Hit, VectorStore

MODES = ("vector", "bm25", "hybrid")


def rrf_fuse(rankings: dict[str, list[Hit]], k: int = 60) -> list[Hit]:
    """score(d) = sum over rankings of 1 / (k + rank). Keeps one Hit per chunk, merging stage scores."""
    fused: dict[tuple[str, int], Hit] = {}
    for ranking in rankings.values():
        for rank, h in enumerate(ranking, 1):
            cur = fused.setdefault(h.key, h)
            if cur is not h:
                cur.scores.update(h.scores)
                cur.distance = cur.distance if cur.distance is not None else h.distance
            cur.scores["rrf"] = cur.scores.get("rrf", 0.0) + 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda h: h.scores["rrf"], reverse=True)


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder, reranker=None, mode: str = "hybrid",
                 candidates: int = 20, rrf_k: int = 60, rerank_pool: int = 20):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.store, self.embedder = store, embedder
        self.reranker = reranker or NoopReranker()
        self.mode, self.candidates, self.rrf_k = mode, candidates, rrf_k
        self.rerank_pool = rerank_pool  # the cross-encoder dominates query latency on CPU
        self._bm25: BM25 | None = None
        self._bm25_chunks: list[Hit] = []
        self._bm25_count = -1

    def _bm25_index(self) -> tuple[BM25, list[Hit]]:
        # Rebuilt only when the store changes size, e.g. after an ingest.
        n = self.store.count()
        if self._bm25 is None or n != self._bm25_count:
            self._bm25_chunks = self.store.all_chunks()
            self._bm25 = BM25([h.text for h in self._bm25_chunks])
            self._bm25_count = n
        return self._bm25, self._bm25_chunks

    def _vector(self, question: str) -> list[Hit]:
        return self.store.query(self.embedder.embed([question])[0], self.candidates)

    def _keyword(self, question: str) -> list[Hit]:
        index, chunks = self._bm25_index()
        out = []
        for i, s in index.top(question, self.candidates):
            c = chunks[i]
            out.append(Hit(c.source, c.index, c.text, None, {"bm25": s}))
        return out

    def retrieve(self, question: str, top_k: int) -> list[Hit]:
        rankings = {}
        if self.mode in ("vector", "hybrid"):
            rankings["vector"] = self._vector(question)
        if self.mode in ("bm25", "hybrid"):
            rankings["bm25"] = self._keyword(question)
        candidates = rrf_fuse(rankings, self.rrf_k)[: max(self.rerank_pool, top_k)]
        return self.reranker.rerank(question, candidates, top_k)


def build_retriever(cfg: dict, store: VectorStore, embedder, reranker=None, mode: str | None = None) -> HybridRetriever:
    r = cfg["retrieval"]
    return HybridRetriever(store, embedder, reranker, mode or r.get("mode", "hybrid"),
                           r.get("candidates", 20), r.get("rrf_k", 60), r.get("rerank_pool", 20))
