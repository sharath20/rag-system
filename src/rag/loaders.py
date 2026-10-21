"""Load documents (PDF, Markdown/text, HTML/web pages) into plain text."""
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class Document:
    source: str  # file path or URL, used in citations
    text: str


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append(f"[page {i}]\n{page.extract_text() or ''}")
    return "\n\n".join(pages)


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def load_file(path: Path) -> Document | None:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _load_pdf(path)
    elif suffix in {".md", ".markdown", ".txt", ".rst"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix in {".html", ".htm"}:
        text = _html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
    else:
        return None
    if not text.strip():
        return None
    return Document(source=path.name, text=text)


def load_url(url: str) -> Document:
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    return Document(source=url, text=_html_to_text(resp.text))


def load_directory(directory: str | Path) -> list[Document]:
    docs = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            doc = load_file(path)
            if doc:
                docs.append(doc)
    return docs
