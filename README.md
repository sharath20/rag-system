# Production-Grade RAG System (Project 1)

Local, citation-first RAG over a public docs sample (FastAPI tutorial). Stack: Ollama (chat + embeddings), ChromaDB, Python.

## Roadmap
- [x] **Phase 1** – ingest (PDF/MD/HTML/URL), 650-token chunks with 100 overlap, Chroma store, top-k retrieval, answers with `[n]` citations, invalid-citation detection
- [x] **Phase 2** – hybrid retrieval (BM25 + vectors, RRF fusion), cross-encoder re-ranker (ONNX), strict citation checks with one repair pass, versioned prompts
- [x] **Phase 3** – 77-item golden set (69 answerable + 8 unanswerable), Ragas faithfulness with a local judge, CI gates for retrieval (live) and faithfulness (committed, fingerprinted report)

## Setup
```bash
pip install -r requirements.txt
ollama pull llama3.2 && ollama pull nomic-embed-text
export PYTHONPATH=src
python -m rag.cli fetch      # download sample docs to data/corpus
python -m rag.cli ingest     # chunk + embed + store
python -m rag.cli ask "How do I add CORS middleware?"
python -m rag.cli retrieve "CORSMiddleware allow_origins" --mode bm25   # inspect retrieval, no LLM
python -m rag.cli ask "..." --prompt answer@1 --no-rerank --lenient     # A/B against Phase 1 behaviour
python -m rag.cli prompts    # list prompt versions + hashes
pytest                       # runs offline with a fake embedder
```

## Evaluation (Phase 3)
```bash
pip install -r requirements-eval.txt     # Ragas + pinned LangChain (only for `eval run`)
python -m rag.cli eval retrieval         # hit rate / MRR / evidence recall, ~5 min, no chat LLM
python -m rag.cli eval run --limit 3     # smoke test the full pipeline + judge
python -m rag.cli eval run               # answers + faithfulness on the 20-item eval.generation_ids subset (~1.5h on CPU, resumable)
python -m rag.cli eval gate              # what CI runs: is the committed report fresh and above thresholds?
```
After changing pipeline code, prompts, config, the corpus or the golden set, re-run `eval run` and commit `eval/reports/latest.json` with the change; otherwise CI fails the PR as stale.

| Metric | Measures | Where |
|---|---|---|
| `hit_rate` | an expected source doc is in the top-k | CI, live |
| `mrr` | how high the first expected source ranks | CI, live |
| `evidence_recall` | the exact evidence sentence is in a retrieved passage | CI, live |
| `faithfulness` | Ragas: share of answer statements supported by the retrieved passages | committed report |
| `answer_rate` | answerable questions that got an answer (stops "refuse everything" gaming faithfulness) | committed report |
| `abstain_rate` | unanswerable questions correctly refused | committed report |

Thresholds live in `config.yaml` under `eval.thresholds`; changing them doesn't invalidate the report.

## Layout
```
config.yaml            models, chunk sizes, retrieval mode, re-ranker, prompt versions, citation strictness
prompts/<name>_vN.txt  versioned prompts (answer_v1, answer_v2, repair_v1); never edited once used
src/rag/loaders.py     PDF / Markdown / HTML / URL -> text
src/rag/chunking.py    token sliding-window chunker
src/rag/embeddings.py  Ollama embed + chat client
src/rag/store.py       ChromaDB wrapper (idempotent upserts)
src/rag/bm25.py        pure-Python Okapi BM25
src/rag/retrieval.py   hybrid retriever: vector + BM25 -> RRF fusion -> re-ranker
src/rag/rerank.py      cross-encoder (ms-marco-MiniLM-L-6-v2 via onnxruntime, no torch)
src/rag/citations.py   strict checks: valid numbers, every sentence cited, lexical support
src/rag/prompts.py     prompt registry ("answer@2" -> file, sha256, metadata)
src/rag/pipeline.py    ingest, answer, strict-mode repair / refusal
src/rag/evaluate.py    golden set, retrieval + generation metrics, fingerprint, CI gate
src/rag/judge.py       Ragas judge on Ollama structured outputs (schema-constrained JSON)
src/rag/cli.py         fetch / ingest / retrieve / ask / prompts / eval
eval/golden.jsonl      golden Q&A: question, reference answer, expected sources, verbatim evidence
eval/reports/          committed eval report (latest.json) checked by CI
.github/workflows/     CI: unit tests, faithfulness gate, live retrieval gate
tests/                 Phase 1-3 tests (offline, fake embedder/reranker/LLM/judge)
```

## Design notes
- Re-ingesting is idempotent (deterministic chunk IDs + upsert).
- Temperature 0 and an explicit "I don't know" fallback to reduce hallucination.
- **Hybrid retrieval**: each retriever returns `candidates` (20) chunks; Reciprocal Rank Fusion merges them by rank, so BM25 and cosine scores never need calibrating. BM25 catches exact identifiers (`CORSMiddleware`, `BackgroundTasks`) that embeddings blur. The BM25 index is built from the Chroma collection in memory and rebuilt when the chunk count changes (fine for docs-sized corpora).
- **Re-ranking**: the cross-encoder scores (question, passage) pairs jointly and picks the final `top_k`. The model (~90MB) downloads from Hugging Face on first use; set `reranker.enabled: false` to skip it.
- **Strict citations** (`citations.strict`): an answer passes only if every citation number exists, every substantive sentence is cited, and at least `min_support` of each sentence's content words appear in its cited passages. Failing answers get one repair attempt (`repair@1`, with the problems listed), then are replaced by the "I don't know" refusal. The support check is lexical, not entailment; Phase 3 adds Ragas faithfulness.
- **Versioned prompts**: every answer records `prompt_id` (e.g. `answer@2#5f32a5f8`, name, version, content hash). Change a prompt by adding `answer_v3.txt`, not by editing v2.
- **Why two gates**: CI runners have no GPU, so answering 77 questions plus judging them would take hours there. Retrieval is measured live in CI (Ollama with only the embedding model). Faithfulness is measured locally, and the report stores a SHA-256 *fingerprint* of everything that can change an answer: pipeline code, prompts, config (minus thresholds and host/timeout), corpus and golden set, with line endings normalized. CI recomputes it and fails the PR if they differ.
- **Judge**: Ragas' own prompts make small models echo the JSON schema instead of filling it, so `judge.py` calls Ollama with `format=<schema>`, which constrains decoding. llama3.2 (3B) is a noisy, strict judge; set `eval.judge_model` to something larger for better scores, and re-run the eval, since the judge model is part of the fingerprint.
- **Golden set integrity**: every answerable item has a verbatim `evidence` quote, and a test asserts it really appears in the listed source, so the set can't silently rot when the corpus changes.
