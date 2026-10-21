"""Token-based sliding-window chunking (default 650 tokens, 100 overlap)."""
from dataclasses import dataclass

import tiktoken

_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        try:
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:  # offline fallback: whitespace "tokens"
            _ENC = False
    return _ENC


def _encode(text: str):
    enc = _encoder()
    return enc.encode(text) if enc else text.split(" ")


def _decode(tokens) -> str:
    enc = _encoder()
    return enc.decode(tokens) if enc else " ".join(tokens)


@dataclass
class Chunk:
    source: str
    index: int
    text: str


def chunk_text(source: str, text: str, chunk_tokens: int = 650, overlap_tokens: int = 100) -> list[Chunk]:
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    tokens = _encode(text)
    step = chunk_tokens - overlap_tokens
    chunks = []
    for i, start in enumerate(range(0, max(len(tokens), 1), step)):
        window = tokens[start : start + chunk_tokens]
        if not window:
            break
        chunks.append(Chunk(source=source, index=i, text=_decode(window).strip()))
        if start + chunk_tokens >= len(tokens):
            break
    return [c for c in chunks if c.text]
