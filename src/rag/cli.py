"""CLI: python -m rag.cli {fetch,ingest,retrieve,ask,prompts,eval}"""
import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .embeddings import client_from_config
from .pipeline import answer_question, ingest_directory
from .prompts import PromptRegistry
from .rerank import NoopReranker, build_reranker
from .retrieval import MODES, build_retriever
from .store import Hit, VectorStore


def _client(cfg):
    return client_from_config(cfg)


def _store(cfg):
    return VectorStore(cfg["store"]["path"], cfg["store"]["collection"])


def _retriever(cfg, args, store, client):
    reranker = NoopReranker() if args.no_rerank else build_reranker(cfg)
    return build_retriever(cfg, store, client, reranker, args.mode)


def _fmt_scores(h: Hit) -> str:
    return " ".join(f"{k}={v:.3f}" for k, v in h.scores.items())


def _add_retrieval_flags(p):
    p.add_argument("--mode", choices=MODES, help="override retrieval.mode")
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    p.add_argument("-k", "--top-k", type=int, help="override retrieval.top_k")


def main(argv=None):
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the public docs sample into the corpus dir")
    sub.add_parser("ingest", help="chunk, embed and store everything in the corpus dir")
    ret = sub.add_parser("retrieve", help="show retrieved passages and scores without calling the LLM")
    ret.add_argument("question")
    _add_retrieval_flags(ret)
    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--prompt", help="prompt ref, e.g. answer@1 (default: prompts.answer)")
    ask.add_argument("--lenient", action="store_true", help="report citation problems but don't refuse")
    ask.add_argument("-v", "--verbose", action="store_true", help="show rejected drafts")
    _add_retrieval_flags(ask)
    sub.add_parser("prompts", help="list prompt versions")

    ev = sub.add_parser("eval", help="evaluate against the golden set (Phase 3)")
    evsub = ev.add_subparsers(dest="eval_cmd", required=True)
    er = evsub.add_parser("retrieval", help="live retrieval metrics (no chat LLM; fast, runs in CI)")
    er.add_argument("--gate", action="store_true", help="exit 1 if below eval.thresholds.retrieval")
    _add_retrieval_flags(er)
    run = evsub.add_parser("run", help="full eval: answers + Ragas faithfulness; writes the report (slow)")
    run.add_argument("--limit", type=int, help="only the first N golden items (smoke test; the gate rejects it)")
    run.add_argument("--fresh", action="store_true", help="discard partial results from an interrupted run")
    evsub.add_parser("rejudge", help="re-score the report's saved answers with the current eval.judge_model "
                                     "(only the judge may have changed since the report was made)")
    evsub.add_parser("gate", help="check the committed report: fresh fingerprint + generation thresholds")
    evsub.add_parser("fingerprint", help="print the current pipeline fingerprint")
    args = p.parse_args(argv)
    cfg = load_config()
    if getattr(args, "top_k", None):
        cfg["retrieval"]["top_k"] = args.top_k

    if args.cmd == "fetch":
        from .corpus import fetch_sample

        n = fetch_sample(cfg["paths"]["corpus_dir"])
        print(f"Downloaded {n} pages to {cfg['paths']['corpus_dir']}")
    elif args.cmd == "ingest":
        store = _store(cfg)
        n_docs, n_chunks = ingest_directory(cfg["paths"]["corpus_dir"], store, _client(cfg), cfg)
        print(f"Ingested {n_docs} documents -> {n_chunks} chunks (store now has {store.count()})")
    elif args.cmd == "retrieve":
        store, client = _store(cfg), _client(cfg)
        hits = _retriever(cfg, args, store, client).retrieve(args.question, cfg["retrieval"]["top_k"])
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {h.source} (chunk {h.index})  {_fmt_scores(h)}")
            print("    " + h.text[:200].replace("\n", " ") + "...\n")
    elif args.cmd == "prompts":
        reg = PromptRegistry(cfg["prompts"]["dir"])
        names = sorted({f.stem.rsplit("_v", 1)[0] for f in reg.dir.glob("*_v*.txt")})
        for name in names:
            for v in reg.versions(name):
                pr = reg.get(f"{name}@{v}")
                active = " (active)" if pr.id.split("#")[0] in (cfg["prompts"]["answer"],
                                                               cfg["citations"].get("repair_prompt")) else ""
                print(f"{pr.id}{active}  {pr.meta.get('description', '')}")
    elif args.cmd == "ask":
        if args.lenient:
            cfg["citations"]["strict"] = False
        store, client = _store(cfg), _client(cfg)
        ans = answer_question(args.question, store, client, cfg,
                              retriever=_retriever(cfg, args, store, client), prompt_ref=args.prompt)
        print(ans.text, "\n")
        print(f"Prompt: {ans.prompt_id}")
        print("Sources:")
        for i, h in enumerate(ans.hits, 1):
            mark = "*" if i in ans.cited and not ans.refused else " "
            print(f" {mark}[{i}] {h.source} (chunk {h.index})  {_fmt_scores(h)}")
        problems = ans.report.problems() if ans.report else []
        if ans.refused:
            print(f"\nREFUSED by strict citation check after {len(ans.drafts)} attempt(s). Last problems:")
        elif problems:
            print("\nCitation problems:")
        for prob in problems:
            print(f"  - {prob}")
        if args.verbose:
            for n, d in enumerate(ans.drafts, 1):
                print(f"\n--- rejected draft {n} ---\n{d}")

    elif args.cmd == "eval":
        sys.exit(_eval(cfg, args))


def _eval(cfg, args) -> int:
    from . import evaluate as E

    golden = E.load_golden(cfg["eval"]["golden"])
    thresholds = cfg["eval"]["thresholds"]
    report_path = Path(cfg["eval"]["report"])
    gen_items = E.generation_items(golden, cfg["eval"].get("generation_ids"))
    gen_ids = [g.id for g in gen_items]

    if args.eval_cmd == "fingerprint":
        print(E.fingerprint())
        return 0

    if args.eval_cmd == "gate":
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        fails = E.gate_report(report, cfg, E.fingerprint(), gen_ids)
        if report and report.get("generation"):
            print(f"Report {report_path.name} ({report['created']}): {report['generation']['metrics']}")
        for f in fails:
            print(f"FAIL: {f}")
        print("Gate: " + ("FAILED" if fails else "passed"))
        return 1 if fails else 0

    store, client = _store(cfg), _client(cfg)
    if store.count() == 0:
        print("Vector store is empty. Run: python -m rag.cli ingest")
        return 1

    if args.eval_cmd == "rejudge":
        return _rejudge(cfg, E, report_path, store)
    top_k = cfg["retrieval"]["top_k"]

    if args.eval_cmd == "retrieval":
        res = E.eval_retrieval(golden, _retriever(cfg, args, store, client), top_k)
        for r in res["rows"]:
            if r["rank"] is None or not r["evidence_found"]:
                print(f"  miss {r['id']}: rank={r['rank']} evidence={r['evidence_found']} got={r['retrieved'][:3]}")
        print(f"Retrieval (top_k={top_k}): {res['metrics']}")
        fails = E.check_thresholds(res["metrics"], thresholds.get("retrieval"), "retrieval")
        for f in fails:
            print(f"FAIL: {f}")
        return 1 if (args.gate and fails) else 0

    # eval run
    items = gen_items[: args.limit] if args.limit else gen_items
    fp = E.fingerprint()
    partial = report_path.parent / f".partial-{fp[:12]}.jsonl"
    if args.fresh and partial.exists():
        partial.unlink()
    retriever = build_retriever(cfg, store, client, build_reranker(cfg))
    print(f"Retrieval eval over {sum(i.answerable for i in golden)} answerable items...")
    retrieval = E.eval_retrieval(golden, retriever, top_k)
    print(f"  {retrieval['metrics']}")
    print(f"Generation eval over {len(items)} items (resumable; partial results in {partial.name})...")
    scorer = E.ragas_faithfulness_scorer(cfg)
    generation = E.eval_generation(
        items, lambda q: answer_question(q, store, client, cfg, retriever=retriever), scorer, partial,
        progress=lambda m: print(m, flush=True))
    report = E.build_report(cfg, golden, retrieval, generation, fp, gen_ids)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nGeneration: {generation['metrics']}\nWrote {report_path}")
    fails = E.gate_report(report, cfg, fp, gen_ids)
    for f in fails:
        print(f"FAIL: {f}")
    return 0


def _rejudge(cfg, E, report_path: Path, store) -> int:
    """Re-score saved answers with the current judge. Answering is the slow part (minutes per
    item on CPU), so swapping the judge shouldn't mean regenerating every answer. Only allowed
    when the report matches the current tree with its own judge put back."""
    import tempfile

    import yaml

    from .config import ROOT

    if not report_path.exists() or not (json.loads(report_path.read_text(encoding="utf-8")).get("generation")):
        print("No report with generation results to re-judge. Run: python -m rag.cli eval run")
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    old_judge, new_judge = report["models"]["judge"], cfg["eval"].get("judge_model", cfg["ollama"]["chat_model"])

    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["eval"]["judge_model"] = old_judge
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        if E.fingerprint(config_path=cfg_path) != report["fingerprint"]:
            print("Refusing to re-judge: more than the judge changed since the report was made. "
                  "Run: python -m rag.cli eval run")
            return 1

    texts = {f"{h.source}#{h.index}": h.text for h in store.all_chunks()}
    questions = {g.id: g.question for g in E.load_golden(cfg["eval"]["golden"])}
    scorer = E.ragas_faithfulness_scorer(cfg)
    rows = report["generation"]["rows"]
    print(f"Re-judging {sum(r['answered'] for r in rows)} answers: {old_judge} -> {new_judge}")
    for i, r in enumerate(rows, 1):
        if not r["answered"]:
            continue
        r.setdefault("faithfulness_by", {})[old_judge] = r["faithfulness"]
        try:
            r["faithfulness"], r["error"] = scorer(questions[r["id"]], r["answer"],
                                                    [texts[s] for s in r["sources"]]), None
        except Exception as exc:
            r["faithfulness"], r["error"] = None, f"faithfulness: {type(exc).__name__}: {exc}"[:500]
        r["faithfulness_by"][new_judge] = r["faithfulness"]
        print(f"[{i}/{len(rows)}] {r['id']}: {r['faithfulness_by'][old_judge]} -> {r['faithfulness']}", flush=True)

    report["generation"]["metrics"] = E.generation_metrics(rows)
    report["models"]["judge"] = new_judge
    report["fingerprint"] = E.fingerprint()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nGeneration: {report['generation']['metrics']}\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    main()
