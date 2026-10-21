"""CLI: python -m rag.cli {fetch,ingest,retrieve,ask,prompts}"""
import argparse

from .config import load_config
from .embeddings import OllamaClient
from .pipeline import answer_question, ingest_directory
from .prompts import PromptRegistry
from .rerank import NoopReranker, build_reranker
from .retrieval import MODES, build_retriever
from .store import Hit, VectorStore


def _client(cfg):
    o = cfg["ollama"]
    return OllamaClient(o["host"], o["embed_model"], o["chat_model"], o.get("timeout", 120))


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


if __name__ == "__main__":
    main()
