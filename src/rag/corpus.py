"""Fetch a small public documentation sample (FastAPI tutorial pages).

Run on your own machine (needs internet): python -m rag.cli fetch
"""
from pathlib import Path

from .loaders import load_url

BASE = "https://fastapi.tiangolo.com/tutorial/"
PAGES = [
    "first-steps/", "path-params/", "query-params/", "body/",
    "query-params-str-validations/", "response-model/", "handling-errors/",
    "dependencies/", "security/first-steps/", "background-tasks/",
    "middleware/", "cors/", "testing/",
]


def fetch_sample(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for page in PAGES:
        url = BASE + page
        try:
            doc = load_url(url)
        except Exception as e:  # keep going if one page fails
            print(f"skip {url}: {e}")
            continue
        name = "fastapi_" + page.strip("/").replace("/", "_") + ".md"
        (out / name).write_text(f"Source: {url}\n\n{doc.text}", encoding="utf-8")
        count += 1
    return count
