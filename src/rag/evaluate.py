"""Phase 3 evaluation: golden set, retrieval metrics, Ragas faithfulness, and the CI gate.

Two kinds of evaluation, because they cost very different amounts:

* retrieval  -- embeddings + BM25 + re-ranker only, no chat LLM. A full pass takes a
                minute, so CI runs it live on every PR.
* generation -- answers every golden question with the chat LLM and scores each answer
                with Ragas faithfulness, using a local judge. On CPU this takes hours, so
                it runs on a developer machine and writes eval/reports/latest.json.

CI cannot re-run generation, so the report carries a *fingerprint*: a hash of everything
that can change an answer (pipeline code, prompts, config, corpus, golden set). The gate
fails if the committed report's fingerprint doesn't match the PR's tree, which means
someone changed the pipeline without re-running the eval, or if any score is below
its threshold.
"""
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .citations import NO_ANSWER
from .config import ROOT

# Pipeline files whose changes can alter answers or scores. cli.py is presentation only.
_FINGERPRINT_EXCLUDE = {"cli.py"}
# Config keys that are machine-specific or don't affect results.
# generation_ids is excluded because the gate checks report coverage against it directly.
_CONFIG_EXCLUDE = {("ollama", "host"), ("ollama", "timeout"), ("eval", "thresholds"), ("eval", "generation_ids")}


@dataclass
class GoldenItem:
    id: str
    question: str
    reference: str
    sources: list[str]
    evidence: str | None
    answerable: bool


def load_golden(path: str | Path) -> list[GoldenItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(GoldenItem(**json.loads(line)))
    return items


def generation_items(golden: list[GoldenItem], ids: list[str] | None) -> list[GoldenItem]:
    """The items the generation eval answers: all of them, or a fixed subset (eval.generation_ids).

    Answering + judging costs minutes per item on CPU, so a hand-picked subset that covers
    every document plus some unanswerable questions keeps a full run to about an hour or two.
    Retrieval is always evaluated on the whole golden set."""
    if not ids:
        return golden
    by_id = {g.id: g for g in golden}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise ValueError(f"eval.generation_ids not in the golden set: {unknown}")
    wanted = set(ids)
    return [g for g in golden if g.id in wanted]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# --- fingerprint ----------------------------------------------------------------

def _file_hash(path: Path) -> str:
    # Normalise line endings so a Windows checkout (CRLF) and CI (LF) agree.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fingerprint(root: Path = ROOT, config_path: Path | None = None, golden_path: Path | None = None) -> str:
    config_path = config_path or root / "config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for section, key in _CONFIG_EXCLUDE:
        raw.get(section, {}).pop(key, None)
    golden_path = golden_path or root / raw["eval"]["golden"]

    parts = {"config": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()}
    files = [p for p in sorted((root / "src" / "rag").glob("*.py")) if p.name not in _FINGERPRINT_EXCLUDE]
    files += sorted((root / raw["prompts"]["dir"]).glob("*.txt"))
    files += sorted((root / raw["paths"]["corpus_dir"]).glob("*"))
    files.append(golden_path)
    for p in files:
        if p.is_file():
            parts[p.relative_to(root).as_posix()] = _file_hash(p)
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


# --- retrieval ------------------------------------------------------------------

def eval_retrieval(golden: list[GoldenItem], retriever, top_k: int) -> dict:
    """hit_rate: a passage from an expected source is in the top_k.
    mrr: mean reciprocal rank of the first such passage.
    evidence_recall: the exact evidence sentence is inside one of the top_k passages."""
    rows = []
    for item in golden:
        if not item.answerable:
            continue
        hits = retriever.retrieve(item.question, top_k)
        rank = next((i for i, h in enumerate(hits, 1) if h.source in item.sources), None)
        ev = normalize(item.evidence or "")
        rows.append({
            "id": item.id,
            "rank": rank,
            "evidence_found": bool(ev) and any(ev in normalize(h.text) for h in hits),
            "retrieved": [f"{h.source}#{h.index}" for h in hits],
        })
    n = len(rows) or 1
    metrics = {
        "n": len(rows),
        "hit_rate": sum(r["rank"] is not None for r in rows) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "evidence_recall": sum(r["evidence_found"] for r in rows) / n,
    }
    return {"metrics": _round(metrics), "rows": rows}


# --- generation -----------------------------------------------------------------

def ragas_faithfulness_scorer(cfg: dict):
    """Returns score(question, answer, contexts) -> float in [0, 1], judged by Ragas.

    Imported lazily so the core package and CI don't need the eval dependencies."""
    import asyncio

    from ragas.metrics.collections import Faithfulness

    from .judge import OllamaJudge

    o, e = cfg["ollama"], cfg["eval"]
    judge = OllamaJudge(o["host"], e.get("judge_model", o["chat_model"]), o.get("num_ctx", 8192),
                        e.get("judge_num_predict", 1024), o.get("timeout", 600))
    metric = Faithfulness(llm=judge)

    def score(question: str, answer: str, contexts: list[str]) -> float:
        return asyncio.run(metric.ascore(user_input=question, response=answer, retrieved_contexts=contexts)).value

    return score


def _generation_row(item: GoldenItem, answer_fn, scorer) -> dict:
    t0 = time.time()
    ans = answer_fn(item.question)
    answered = ans.text.strip() != NO_ANSWER
    row = {
        "id": item.id,
        "answerable": item.answerable,
        "answered": answered,
        "refused_by_strict_check": ans.refused,
        "decision_correct": answered == item.answerable,
        "answer": ans.text,
        "cited": ans.cited,
        "prompt_id": ans.prompt_id,
        "sources": [f"{h.source}#{h.index}" for h in ans.hits],
        "faithfulness": None,
        "error": None,
    }
    if answered and scorer is not None:
        try:
            row["faithfulness"] = scorer(item.question, ans.text, [h.text for h in ans.hits])
        except Exception as exc:  # a judge failure shouldn't throw away an hour of answers
            row["error"] = f"faithfulness: {type(exc).__name__}: {exc}"[:500]
    row["seconds"] = round(time.time() - t0, 1)
    return row


def generation_metrics(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    scored = [r["faithfulness"] for r in rows
              if r["faithfulness"] is not None and not math.isnan(r["faithfulness"])]
    return _round({
        "n": len(rows),
        # Faithfulness: share of answer statements the judge found supported by the retrieved passages.
        "faithfulness": sum(scored) / len(scored) if scored else 0.0,
        "faithfulness_n": len(scored),
        # Answer rate: answerable questions that got a (non-refusal) answer. Guards against a
        # system that "wins" faithfulness by refusing everything.
        "answer_rate": sum(r["answered"] for r in answerable) / (len(answerable) or 1),
        # Abstain rate: unanswerable questions correctly refused instead of hallucinated.
        "abstain_rate": sum(not r["answered"] for r in unanswerable) / (len(unanswerable) or 1),
        "judge_errors": sum(bool(r["error"]) for r in rows),
    })


def eval_generation(golden: list[GoldenItem], answer_fn, scorer, partial_path: Path | None = None,
                    progress=print) -> dict:
    """Answers and scores every golden item. Rows are appended to partial_path as they finish,
    so an interrupted multi-hour run resumes where it stopped."""
    done: dict[str, dict] = {}
    if partial_path and partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    rows = []
    for i, item in enumerate(golden, 1):
        if item.id in done:
            rows.append(done[item.id])
            continue
        row = _generation_row(item, answer_fn, scorer)
        rows.append(row)
        if partial_path:
            with open(partial_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f_txt = "-" if row["faithfulness"] is None else f"{row['faithfulness']:.2f}"
        progress(f"[{i}/{len(golden)}] {item.id}: answered={row['answered']} "
                 f"correct={row['decision_correct']} faithfulness={f_txt} ({row['seconds']}s)")
    return {"metrics": generation_metrics(rows), "rows": rows}


# --- report + gate ----------------------------------------------------------------

def build_report(cfg: dict, golden: list[GoldenItem], retrieval: dict, generation: dict | None, fp: str,
                 generation_ids: list[str] | None = None) -> dict:
    o = cfg["ollama"]
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": fp,
        "golden_size": len(golden),
        "generation_ids": generation_ids or [g.id for g in golden],
        "models": {
            "chat": o["chat_model"], "embed": o["embed_model"],
            "judge": cfg["eval"].get("judge_model", o["chat_model"]),
            "reranker": cfg["reranker"]["model"] if cfg["reranker"].get("enabled") else None,
        },
        "prompt": cfg["prompts"]["answer"],
        "retrieval_mode": cfg["retrieval"].get("mode", "hybrid"),
        "retrieval": retrieval,
        "generation": generation,
    }


def check_thresholds(metrics: dict, thresholds: dict, label: str) -> list[str]:
    fails = []
    for name, minimum in (thresholds or {}).items():
        value = metrics.get(name)
        if value is None:
            fails.append(f"{label}.{name}: missing from results")
        elif value < minimum:
            fails.append(f"{label}.{name} = {value:.3f} is below the threshold {minimum:.3f}")
    return fails


def gate_report(report: dict | None, cfg: dict, current_fingerprint: str, expected_ids: list[str]) -> list[str]:
    """Problems that should fail CI. An empty list means the gate passes."""
    if report is None:
        return ["No eval report found. Run: python -m rag.cli eval run"]
    fails = []
    if report.get("fingerprint") != current_fingerprint:
        fails.append("Eval report is stale: pipeline code, prompts, config, corpus or golden set changed "
                     "since it was generated. Re-run: python -m rag.cli eval run")
    gen = report.get("generation")
    if not gen:
        fails.append("Eval report has no generation results (was it run with --retrieval-only?)")
    else:
        covered = {r["id"] for r in gen.get("rows", [])}
        missing = [i for i in expected_ids if i not in covered]
        if missing:
            fails.append(f"Eval report is missing {len(missing)}/{len(expected_ids)} expected items "
                         f"(run with --limit, or eval.generation_ids changed?): {missing[:5]}")
        fails += check_thresholds(gen["metrics"], cfg["eval"]["thresholds"].get("generation"), "generation")
    return fails


def _round(d: dict) -> dict:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
