"""ChromaDB vector store wrapper."""
import hashlib
from dataclasses import dataclass, field

import chromadb

from .chunking import Chunk


@dataclass
class Hit:
    source: str
    index: int
    text: str
    distance: float | None = None  # cosine distance from the vector search, if it found this chunk
    scores: dict[str, float] = field(default_factory=dict)  # per-stage scores: vector, bm25, rrf, rerank

    @property
    def key(self) -> tuple[str, int]:
        return (self.source, self.index)


class VectorStore:
    def __init__(self, path: str, collection: str):
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _id(chunk: Chunk) -> str:
        return hashlib.sha1(f"{chunk.source}:{chunk.index}".encode()).hexdigest()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        # upsert makes re-ingesting the same corpus idempotent
        self.col.upsert(
            ids=[self._id(c) for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, "index": c.index} for c in chunks],
        )

    def query(self, embedding: list[float], top_k: int) -> list[Hit]:
        n = min(top_k, self.count())
        if n == 0:
            return []
        res = self.col.query(query_embeddings=[embedding], n_results=n)
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(meta["source"], meta["index"], doc, dist, {"vector": 1 - dist}))
        return hits

    def all_chunks(self) -> list[Hit]:
        """Every stored chunk, for building the BM25 index."""
        res = self.col.get(include=["documents", "metadatas"])
        return [Hit(m["source"], m["index"], d) for d, m in zip(res["documents"], res["metadatas"])]

    def count(self) -> int:
        return self.col.count()
