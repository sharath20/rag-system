"""Phase 3: golden set integrity, retrieval metrics, generation metrics, fingerprint and CI gate. Offline."""
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import evaluate as E  # noqa: E402
from rag.citations import NO_ANSWER  # noqa: E402
from rag.config import load_config  # noqa: E402
from rag.pipeline import Answer  # noqa: E402
from rag.store import Hit  # noqa: E402

GOLDEN = E.load_golden(ROOT / "eval" / "golden.jsonl")


# --- golden set integrity ------------------------------------------------------

def test_golden_set_size_and_ids():
    assert 50 <= len(GOLDEN) <= 200
    assert len({g.id for g in GOLDEN}) == len(GOLDEN)
    assert sum(not g.answerable for g in GOLDEN) >= 5  # refusal behaviour is measured too


def test_golden_evidence_is_verbatim_in_its_source():
    corpus = ROOT / "data" / "corpus"
    texts = {p.name: E.normalize(p.read_text(encoding="utf-8")) for p in corpus.glob("*.md")}
    for g in GOLDEN:
        if not g.answerable:
            assert g.sources == [] and g.evidence is None and g.reference == NO_ANSWER
            continue
        assert g.sources and all(s in texts for s in g.sources), g.id
        assert any(E.normalize(g.evidence) in texts[s] for s in g.sources), f"{g.id}: evidence not in source"


def test_every_corpus_doc_is_covered():
    covered = {s for g in GOLDEN for s in g.sources}
    assert covered == {p.name for p in (ROOT / "data" / "corpus").glob("*.md")}


# --- retrieval metrics -------------------------------------------------------------

ITEMS = [
    E.GoldenItem("a", "q-a", "r", ["a.md"], "alpha fact", True),
    E.GoldenItem("b", "q-b", "r", ["b.md"], "beta fact", True),
    E.GoldenItem("none", "q-none", NO_ANSWER, [], None, False),
]


class TableRetriever:
    def __init__(self, table):
        self.table = table

    def retrieve(self, question, top_k):
        return self.table[question][:top_k]


def test_retrieval_metrics():
    r = TableRetriever({
        "q-a": [Hit("a.md", 0, "the ALPHA   fact is here")],                       # rank 1, evidence found
        "q-b": [Hit("x.md", 0, "noise"), Hit("b.md", 3, "unrelated part of b")],  # rank 2, no evidence
    })
    res = E.eval_retrieval(ITEMS, r, top_k=5)
    assert res["metrics"] == {"n": 2, "hit_rate": 1.0, "mrr": 0.75, "evidence_recall": 0.5}


# --- generation metrics -------------------------------------------------------------

def _answer(text):
    return Answer(text=text, hits=[Hit("a.md", 0, "ctx")], cited=[1], invalid_citations=[], prompt_id="answer@2#x")


def test_generation_metrics_and_resume(tmp_path):
    replies = {"q-a": "Alpha is a fact [1].", "q-b": NO_ANSWER, "q-none": "Made-up answer [1]."}
    calls = []

    def answer_fn(q):
        calls.append(q)
        return _answer(replies[q])

    partial = tmp_path / "partial.jsonl"
    res = E.eval_generation(ITEMS, answer_fn, lambda q, a, c: 0.8, partial, progress=lambda m: None)
    m = res["metrics"]
    assert m["faithfulness"] == 0.8 and m["faithfulness_n"] == 2  # refusals aren't scored
    assert m["answer_rate"] == 0.5   # q-b wrongly refused
    assert m["abstain_rate"] == 0.0  # q-none hallucinated
    assert [r["decision_correct"] for r in res["rows"]] == [True, False, False]

    # A second run resumes from the partial file without calling the LLM again.
    calls.clear()
    again = E.eval_generation(ITEMS, answer_fn, lambda q, a, c: 0.0, partial, progress=lambda m: None)
    assert calls == [] and again["metrics"] == m


def test_judge_failure_is_recorded_not_fatal(tmp_path):
    def boom(q, a, c):
        raise RuntimeError("judge down")

    res = E.eval_generation(ITEMS[:1], lambda q: _answer("Alpha [1]."), boom, None, progress=lambda m: None)
    assert res["rows"][0]["faithfulness"] is None and "judge down" in res["rows"][0]["error"]
    assert res["metrics"]["judge_errors"] == 1


# --- fingerprint + gate ----------------------------------------------------------------

@pytest.fixture
def repo_copy(tmp_path):
    for d in ("src", "prompts", "data/corpus", "eval"):
        shutil.copytree(ROOT / d, tmp_path / d, ignore=shutil.ignore_patterns("__pycache__", "reports"))
    shutil.copy(ROOT / "config.yaml", tmp_path / "config.yaml")
    return tmp_path


def test_fingerprint_tracks_pipeline_but_not_thresholds_or_line_endings(repo_copy):
    base = E.fingerprint(repo_copy)

    cfg = repo_copy / "config.yaml"
    cfg.write_text(re.sub(r"faithfulness: [0-9.]+", "faithfulness: 0.99", cfg.read_text()))
    assert E.fingerprint(repo_copy) == base  # thresholds don't invalidate a report

    prompt = repo_copy / "prompts" / "answer_v2.txt"
    prompt.write_bytes(prompt.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert E.fingerprint(repo_copy) == base  # CRLF checkout == LF checkout

    prompt.write_text(prompt.read_text() + "\nBe concise.\n")
    assert E.fingerprint(repo_copy) != base  # prompt change -> stale report


def test_fingerprint_changes_with_code_and_golden(repo_copy):
    base = E.fingerprint(repo_copy)
    (repo_copy / "src" / "rag" / "retrieval.py").write_text("# changed\n", encoding="utf-8")
    after_code = E.fingerprint(repo_copy)
    assert after_code != base
    with open(repo_copy / "eval" / "golden.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "new", "question": "q", "reference": "r", "sources": [],
                            "evidence": None, "answerable": False}) + "\n")
    assert E.fingerprint(repo_copy) != after_code


IDS = [g.id for g in GOLDEN]


def _report(fp, ids=IDS, **metrics):
    m = {"n": len(ids), "faithfulness": 0.9, "answer_rate": 0.9, "abstain_rate": 0.9, **metrics}
    return {"fingerprint": fp, "generation": {"metrics": m, "rows": [{"id": i} for i in ids]}}


def test_gate_passes_fresh_good_report():
    cfg = load_config()
    assert E.gate_report(_report("fp"), cfg, "fp", IDS) == []


def test_gate_fails_stale_low_partial_or_missing_reports():
    cfg = load_config()
    assert "No eval report" in E.gate_report(None, cfg, "fp", IDS)[0]
    assert "stale" in E.gate_report(_report("old"), cfg, "new", IDS)[0]
    low = E.gate_report(_report("fp", faithfulness=0.1), cfg, "fp", IDS)
    assert len(low) == 1 and "generation.faithfulness" in low[0]
    partial = E.gate_report(_report("fp", ids=IDS[:5]), cfg, "fp", IDS)
    assert f"missing {len(IDS) - 5}/{len(IDS)}" in partial[0]


def test_generation_subset_is_valid_and_covers_every_doc():
    cfg = load_config()
    items = E.generation_items(GOLDEN, cfg["eval"].get("generation_ids"))
    covered = {s for g in items for s in g.sources}
    assert covered >= {p.name for p in (ROOT / "data" / "corpus").glob("*.md")}
    assert any(not g.answerable for g in items)
    assert E.generation_items(GOLDEN, None) == GOLDEN
    with pytest.raises(ValueError):
        E.generation_items(GOLDEN, ["no-such-id"])


def test_changing_the_subset_does_not_change_fingerprint(repo_copy):
    base = E.fingerprint(repo_copy)
    cfg = repo_copy / "config.yaml"
    cfg.write_text(cfg.read_text().replace("generation_ids: [bg-01,", "generation_ids: [bg-02,"))
    assert E.fingerprint(repo_copy) == base  # coverage is checked by the gate instead
