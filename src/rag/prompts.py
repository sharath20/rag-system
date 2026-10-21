"""Versioned prompt registry.

Prompts live in prompts/<name>_v<version>.txt and are referenced as "name@version"
(e.g. "answer@2"). Files are never edited in place once used: a change becomes a
new version, so every answer can be traced to the exact prompt that produced it.

A file may start with a metadata block:

    ---
    description: why this version exists
    ---
"""
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

_FILE = re.compile(r"^(?P<name>[a-z0-9_]+)_v(?P<version>\d+)\.txt$")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    template: str
    sha256: str
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable identifier recorded with every answer, e.g. answer@2#1a2b3c4d."""
        return f"{self.name}@{self.version}#{self.sha256[:8]}"

    def render(self, **kwargs) -> str:
        return self.template.format(**kwargs)


def _parse(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    header, _, body = text[4:].partition("\n---\n")
    meta = {}
    for line in header.splitlines():
        k, sep, v = line.partition(":")
        if sep:
            meta[k.strip()] = v.strip()
    return meta, body.lstrip("\n")


class PromptRegistry:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def versions(self, name: str) -> list[int]:
        out = []
        for p in self.dir.glob(f"{name}_v*.txt"):
            m = _FILE.match(p.name)
            if m and m["name"] == name:
                out.append(int(m["version"]))
        return sorted(out)

    def get(self, ref: str) -> Prompt:
        """ref is "name@version", or "name"/"name@latest" for the highest version."""
        name, _, ver = ref.partition("@")
        if ver in ("", "latest"):
            versions = self.versions(name)
            if not versions:
                raise FileNotFoundError(f"no prompt named {name!r} in {self.dir}")
            version = versions[-1]
        else:
            version = int(ver)
        path = self.dir / f"{name}_v{version}.txt"
        if not path.exists():
            raise FileNotFoundError(f"prompt {name}@{version} not found ({path})")
        raw = path.read_text(encoding="utf-8")
        meta, template = _parse(raw)
        return Prompt(name, version, template, hashlib.sha256(raw.encode()).hexdigest(), meta)
