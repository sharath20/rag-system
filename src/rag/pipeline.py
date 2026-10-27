"""Ingest + retrieve + answer-with-citations pipeline."""
from dataclasses import dataclass, field

from .chunking import chunk_text
from .citations import NO_ANSWER, CitationReport, check_citations, extract_citations
from .loaders import Document, load_directory
from .prompts import PromptRegistry
from .rerank import build_reranker
from .retrieval import HybridRetriever, build_retriever
from .store import Hit, VectorStore

__all__ = ["NO_ANSWER", "Answer", "answer_question", "extract_citations", "format_context",
           "ingest_directory", "ingest_documents"]


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    cited: list[int]  # 1-based passage numbers actually cited by the model
    invalid_citations: list[int]  # cited numbers that don't exist in the context
    report: CitationReport | None = None
    prompt_id: str = ""  # e.g. answer@2#1a2b3c4d, for tracing an answer back to its prompt
    refused: bool = False  # True when strict mode replaced a failing answer with NO_ANSWER
    drafts: list[str] = field(default_factory=list)  # rejected attempts, kept for debugging


def ingest_documents(docs: list[Document], store: VectorStore, embedder, cfg: dict, batch: int = 32) -> int:
    ch = cfg["chunking"]
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc.source, doc.text, ch["chunk_tokens"], ch["overlap_tokens"]))
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        store.add(part, embedder.embed([c.text for c in part]))
    return len(chunks)


def ingest_directory(directory: str, store: VectorStore, embedder, cfg: dict) -> tuple[int, int]:
    docs = load_directory(directory)
    return len(docs), ingest_documents(docs, store, embedder, cfg)


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] (source: {h.source}, chunk {h.index})\n{h.text}" for i, h in enumerate(hits, 1))


def answer_question(question: str, store: VectorStore, client, cfg: dict,
                    retriever: HybridRetriever | None = None, prompt_ref: str | None = None) -> Answer:
    if retriever is None:
        retriever = build_retriever(cfg, store, client, build_reranker(cfg))
    hits = retriever.retrieve(question, cfg["retrieval"]["top_k"])

    registry = PromptRegistry(cfg["prompts"]["dir"])
    prompt = registry.get(prompt_ref or cfg["prompts"]["answer"])
    context = format_context(hits)
    cc = cfg.get("citations", {})
    strict, min_support = cc.get("strict", True), cc.get("min_support", 0.3)
    uncited_min_support = cc.get("uncited_min_support", 0.6)

    def check(text: str) -> CitationReport:
        return check_citations(text, [h.text for h in hits], min_support, uncited_min_support=uncited_min_support)

    text = client.chat(prompt.render(context=context, question=question)).strip()
    report = check(text)
    drafts: list[str] = []

    if strict:
        repair = registry.get(cc.get("repair_prompt", "repair@latest"))
        for _ in range(cc.get("max_repairs", 1)):
            if report.ok:
                break
            drafts.append(text)
            text = client.chat(repair.render(
                context=context, question=question, draft=text,
                problems="\n".join(f"- {p}" for p in report.problems()),
            )).strip()
            report = check(text)

    refused = strict and not report.ok
    if refused:
        drafts.append(text)
    return Answer(text=NO_ANSWER if refused else text, hits=hits, cited=report.cited,
                  invalid_citations=report.invalid, report=report, prompt_id=prompt.id,
                  refused=refused, drafts=drafts)
