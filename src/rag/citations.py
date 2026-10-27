"""Strict citation checks.

An answer passes only if:
  1. every citation number points at a retrieved passage,
  2. every substantive sentence carries at least one citation, unless it is a
     summary that the passages cited elsewhere in the answer clearly support
     (e.g. "Yes, X can be Y. The docs say 'X can be Y' [1]."), and
  3. each cited sentence is lexically supported by its cited passages
     (enough of its content words appear in them).

Check 3 is a cheap heuristic that catches citations attached to the wrong
passage. It is not an entailment check; Phase 3 adds Ragas faithfulness for that.
"""
import re
from dataclasses import dataclass, field

from .bm25 import tokenize

NO_ANSWER = "I don't know based on the provided documents."

_CITE = re.compile(r"\[(\d+)\]")
_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_LEADING_CITES = re.compile(r"^((?:\[\d+\])+)\s*")
_STOP = set("""
a an the and or but if then else of to in on at by for with from as is are was were be been being
it its this that these those you your we our they their he she his her them can could should would
will may might must do does did not no yes so than too very just also only use used using which
what when where who how why there here into about over under more most such any all each some
""".split())


@dataclass
class Unsupported:
    sentence: str
    cited: list[int]
    support: float  # fraction of content words found in the cited passages


@dataclass
class CitationReport:
    cited: list[int]
    invalid: list[int]
    uncited: list[str] = field(default_factory=list)
    unsupported: list[Unsupported] = field(default_factory=list)
    is_refusal: bool = False

    @property
    def ok(self) -> bool:
        return self.is_refusal or not (self.invalid or self.uncited or self.unsupported)

    def problems(self) -> list[str]:
        """Human-readable issues, also fed back to the model in the repair prompt."""
        out = [f"Citation [{n}] does not exist; only the numbered passages may be cited." for n in self.invalid]
        out += [f'Sentence has no citation: "{s}"' for s in self.uncited]
        out += [
            f'Sentence is not supported by the passage(s) it cites {u.cited}: "{u.sentence}"'
            for u in self.unsupported
        ]
        if not self.is_refusal and not self.cited:
            out.append("The answer cites no passages at all.")
        return out


def extract_citations(text: str) -> list[int]:
    return sorted({int(n) for n in _CITE.findall(text)})


def _content_words(text: str) -> set[str]:
    return {_stem(w) for w in tokenize(_CITE.sub(" ", text)) if len(w) > 2 and w not in _STOP}


def _stem(w: str) -> str:
    # Crude plural/verb folding so "dependencies"/"dependency" and "runs"/"run" match.
    for suf in ("ies", "es", "s", "ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for part in _SENTENCE_SPLIT.split(_CODE_BLOCK.sub(" ", text)):
        part = part.strip()
        # A citation placed after the period ("... server. [1] Next") belongs to the previous sentence.
        m = _LEADING_CITES.match(part)
        if m and out:
            out[-1] += " " + m[1]
            part = part[m.end():]
        if part:
            out.append(part)
    return out


def check_citations(answer: str, passages: list[str], min_support: float = 0.3,
                    min_words: int = 4, uncited_min_support: float = 0.6) -> CitationReport:
    cited = extract_citations(answer)
    invalid = [n for n in cited if n < 1 or n > len(passages)]
    if answer.strip().rstrip(".") == NO_ANSWER.rstrip("."):
        return CitationReport(cited, invalid, is_refusal=True)

    report = CitationReport(cited, invalid)
    passage_words = [_content_words(p) for p in passages]
    valid = [n for n in cited if 1 <= n <= len(passages)]
    answer_cited_words = set().union(*(passage_words[n - 1] for n in valid)) if valid else set()
    for s in sentences(answer):
        words = _content_words(s)
        refs = [n for n in extract_citations(s) if 1 <= n <= len(passages)]
        if len(words) < min_words or s.endswith(":"):
            continue  # connective text, list lead-ins, headings
        if not _CITE.search(s):
            # An uncited sentence passes only with a higher bar than a cited one, measured against
            # the passages the answer does cite: a restated claim, not a new one.
            support = len(words & answer_cited_words) / len(words)
            if support < uncited_min_support:
                report.uncited.append(s)
            continue
        if not refs:
            continue  # only invalid citations; already reported
        supported = set().union(*(passage_words[n - 1] for n in refs))
        support = len(words & supported) / len(words)
        if support < min_support:
            report.unsupported.append(Unsupported(s, refs, round(support, 2)))
    return report
