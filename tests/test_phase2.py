"""Phase 2: hybrid retrieval, re-ranking, strict citations, versioned prompts. Runs offline."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.bm25 import BM25  # noqa: E402
from rag.citations import NO_ANSWER, check_citations, sentences  # noqa: E402
from rag.config import load_config  # noqa: E402
from rag.loaders import Document  # noqa: E402
from rag.pipeline import answer_question, ingest_documents  # noqa: E402
from rag.prompts import PromptRegistry  # noqa: E402
from rag.retrieval import HybridRetriever, rrf_fuse  # noqa: E402
from rag.store import Hit, VectorStore  # noqa: E402
from test_pipeline import FakeClient  # noqa: E402

DOCS = [
    Document("uvicorn.md", "uvicorn is the asgi server that runs fastapi applications in production"),
    Document("cors.md", "add CORSMiddleware with allow_origins to control which browser origins may call your api"),
    Document("tests.md", "testclient lets you write tests for endpoints without running a server"),
    Document("bg.md", "BackgroundTasks run a function after returning the response, like sending an email"),
]


class ScriptedClient(FakeClient):
    """FakeClient whose chat replies come from a list, recording every prompt it receives."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, prompt, temperature=0.0):
        self.prompts.append(prompt)
        return self.replies.pop(0)


class ReverseReranker:
    """Stand-in cross-encoder: proves the retriever applies the re-ranker's order."""

    def rerank(self, query, hits, top_k):
        for i, h in enumerate(hits):
            h.scores["rerank"] = -i
        return list(reversed(hits))[:top_k]


@pytest.fixture
def setup(tmp_path):
    cfg = load_config()
    cfg["reranker"]["enabled"] = False
    store = VectorStore(str(tmp_path / "chroma"), "phase2")
    ingest_documents(DOCS, store, FakeClient(), cfg)
    return cfg, store


# --- BM25 + fusion ---------------------------------------------------------

def test_bm25_prefers_exact_term_and_ignores_misses():
    bm = BM25([d.text for d in DOCS])
    top = bm.top("CORSMiddleware allow_origins", 4)
    assert top[0][0] == 1
    assert len(top) == 1  # docs with no matching term are dropped


def test_rrf_merges_duplicates_and_rewards_agreement():
    a, b, c = Hit("a", 0, "A", 0.1, {"vector": 0.9}), Hit("b", 0, "B"), Hit("c", 0, "C")
    fused = rrf_fuse({"vector": [a, b], "bm25": [Hit("c", 0, "C", None, {"bm25": 3.0}), Hit("b", 0, "B")]
                      }, k=60)
    assert [h.source for h in fused][0] == "b"  # ranked by both retrievers beats rank-1 in only one
    assert len(fused) == 3
    assert c.key in {h.key for h in fused}


def test_hybrid_finds_keyword_match_vector_misses(setup):
    cfg, store = setup
    q = "BackgroundTasks"
    bm25_only = HybridRetriever(store, FakeClient(), mode="bm25").retrieve(q, 1)
    hybrid = HybridRetriever(store, FakeClient(), mode="hybrid").retrieve(q, 4)
    assert bm25_only[0].source == "bg.md"
    assert "bg.md" in [h.source for h in hybrid]
    hit = next(h for h in hybrid if h.source == "bg.md")
    assert {"bm25", "rrf"} <= hit.scores.keys()


def test_reranker_controls_final_order(setup):
    _, store = setup
    base = HybridRetriever(store, FakeClient()).retrieve("asgi server fastapi", 4)
    reranked = HybridRetriever(store, FakeClient(), reranker=ReverseReranker()).retrieve("asgi server fastapi", 4)
    assert [h.key for h in reranked] == [h.key for h in reversed(base)]


def test_bm25_index_rebuilds_after_ingest(setup):
    cfg, store = setup
    r = HybridRetriever(store, FakeClient(), mode="bm25")
    assert r.retrieve("pydantic", 1) == []
    ingest_documents([Document("models.md", "pydantic models validate request bodies")], store, FakeClient(), cfg)
    assert r.retrieve("pydantic", 1)[0].source == "models.md"


def test_bad_mode_rejected(setup):
    with pytest.raises(ValueError):
        HybridRetriever(setup[1], FakeClient(), mode="fuzzy")


# --- strict citations -------------------------------------------------------

PASSAGES = [d.text for d in DOCS]


def test_citation_report_passes_good_answer():
    r = check_citations("Uvicorn is the ASGI server that runs FastAPI applications [1].", PASSAGES)
    assert r.ok and r.cited == [1]


def test_citation_report_flags_uncited_sentence():
    r = check_citations("Uvicorn runs FastAPI applications [1]. It also compiles Rust code for speed.", PASSAGES)
    assert not r.ok
    assert r.uncited == ["It also compiles Rust code for speed."]


def test_uncited_summary_backed_by_cited_passage_passes():
    # Real llama3.2 answer from the Phase 3 eval (bg-03) that strict mode used to refuse.
    passages = ["It can be an async def or normal def function, FastAPI will know how to handle it correctly. "
                "Create a function to be run as the background task."]
    answer = ("A background task function can be a normal `def` instead of `async def`. According to the "
              "documentation, \"It can be an `async def` or normal `def` function, FastAPI will know how to "
              "handle it correctly.\" [1]")
    assert check_citations(answer, passages).ok


def test_uncited_new_claim_still_fails_even_next_to_a_good_citation():
    answer = ("Uvicorn is the ASGI server that runs FastAPI applications [1]. "
              "Gunicorn workers should equal twice the number of CPU cores.")
    r = check_citations(answer, PASSAGES)
    assert not r.ok and r.uncited == ["Gunicorn workers should equal twice the number of CPU cores."]


def test_citation_report_flags_wrong_passage():
    r = check_citations("Uvicorn is the ASGI server that runs FastAPI applications [3].", PASSAGES)
    assert not r.ok and r.unsupported[0].cited == [3]


def test_citation_report_flags_invalid_number():
    r = check_citations("Uvicorn runs FastAPI applications in production [7].", PASSAGES)
    assert r.invalid == [7] and not r.ok


def test_refusal_and_code_blocks_are_exempt():
    assert check_citations(NO_ANSWER, PASSAGES).ok
    text = "Add CORSMiddleware with allow_origins [2]:\n```python\napp.add_middleware(CORSMiddleware)\n```"
    assert check_citations(text, PASSAGES).ok


def test_trailing_citation_stays_with_its_sentence():
    assert sentences("Uvicorn runs apps. [1] Tests use TestClient [3].") == [
        "Uvicorn runs apps. [1]", "Tests use TestClient [3]."]


def test_strict_mode_repairs_bad_answer(setup):
    cfg, store = setup
    client = ScriptedClient([
        "Uvicorn runs FastAPI applications. Also see [9].",
        "Uvicorn is the ASGI server that runs FastAPI applications in production [1].",
    ])
    ans = answer_question("which asgi server runs fastapi applications", store, client, cfg)
    assert not ans.refused and ans.cited == [1]
    assert len(ans.drafts) == 1 and len(client.prompts) == 2
    assert "Citation [9] does not exist" in client.prompts[1]  # problems were fed back


def test_strict_mode_refuses_when_repair_fails(setup):
    cfg, store = setup
    client = ScriptedClient(["Uvicorn compiles Rust [9].", "Still uncited prose about servers here."])
    ans = answer_question("which asgi server runs fastapi applications", store, client, cfg)
    assert ans.refused and ans.text == NO_ANSWER
    assert len(ans.drafts) == 2


# --- versioned prompts --------------------------------------------------------

def test_prompt_registry_versions_and_ids(tmp_path):
    (tmp_path / "answer_v1.txt").write_text("Q: {question}")
    (tmp_path / "answer_v2.txt").write_text("---\ndescription: newer\n---\nQ2: {question}")
    reg = PromptRegistry(tmp_path)
    assert reg.versions("answer") == [1, 2]
    latest = reg.get("answer@latest")
    assert latest.version == 2 and latest.meta["description"] == "newer"
    assert latest.render(question="hi") == "Q2: hi"
    assert reg.get("answer@1").id.startswith("answer@1#")
    with pytest.raises(FileNotFoundError):
        reg.get("answer@3")


def test_answer_records_prompt_version(setup):
    cfg, store = setup
    client = ScriptedClient(["Uvicorn is the ASGI server that runs FastAPI applications [1]."] * 2)
    assert answer_question("asgi server", store, client, cfg).prompt_id.startswith("answer@2#")
    assert answer_question("asgi server", store, client, cfg, prompt_ref="answer@1").prompt_id.startswith("answer@1#")


def test_repo_prompts_render():
    reg = PromptRegistry(load_config()["prompts"]["dir"])
    for ref in ("answer@1", "answer@2"):
        assert "Q?" in reg.get(ref).render(context="C", question="Q?")
    assert "P!" in reg.get("repair@1").render(context="C", question="Q", draft="D", problems="P!")
