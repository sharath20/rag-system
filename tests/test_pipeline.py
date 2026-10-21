import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunking import chunk_text  # noqa: E402
from rag.config import load_config  # noqa: E402
from rag.loaders import Document  # noqa: E402
from rag.pipeline import answer_question, extract_citations, ingest_documents  # noqa: E402
from rag.store import VectorStore  # noqa: E402


class FakeClient:
    """Deterministic bag-of-words embedder + canned chat, no Ollama needed."""

    DIM = 64

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
            out.append(v)
        return out

    def chat(self, prompt, temperature=0.0):
        return "Uvicorn serves the app [1]. Also see [9]."


def test_chunk_sizes_and_overlap():
    text = " ".join(f"w{i}" for i in range(2000))
    chunks = chunk_text("a.md", text, chunk_tokens=300, overlap_tokens=50)
    assert len(chunks) > 3
    assert all(c.text for c in chunks)
    # consecutive chunks share overlapping content
    first_tail = set(chunks[0].text.split()[-20:])
    assert first_tail & set(chunks[1].text.split())


def test_overlap_validation():
    try:
        chunk_text("a", "x y z", chunk_tokens=10, overlap_tokens=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_extract_citations():
    assert extract_citations("a [2] b [1][2] c") == [1, 2]


def test_end_to_end_retrieval_and_invalid_citation(tmp_path):
    cfg = load_config()
    cfg["store"]["path"] = str(tmp_path / "chroma")
    cfg["reranker"]["enabled"] = False  # no model download in unit tests
    cfg["citations"]["strict"] = False  # Phase 1 behaviour: report, don't refuse
    store = VectorStore(cfg["store"]["path"], "test")
    client = FakeClient()
    docs = [
        Document("uvicorn.md", "uvicorn is the asgi server that runs fastapi applications in production"),
        Document("cors.md", "cors middleware controls which browser origins may call your api"),
        Document("tests.md", "testclient lets you write tests for endpoints without running a server"),
    ]
    n = ingest_documents(docs, store, client, cfg)
    assert n == 3 and store.count() == 3

    # re-ingest is idempotent
    ingest_documents(docs, store, client, cfg)
    assert store.count() == 3

    ans = answer_question("which asgi server runs fastapi applications", store, client, cfg)
    assert ans.hits[0].source == "uvicorn.md"
    assert ans.cited == [1, 9]
    assert ans.invalid_citations == [9]
