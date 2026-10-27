from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against the project root, not the cwd.
    cfg["store"]["path"] = str((ROOT / cfg["store"]["path"]).resolve())
    cfg["paths"]["corpus_dir"] = str((ROOT / cfg["paths"]["corpus_dir"]).resolve())
    cfg["prompts"]["dir"] = str((ROOT / cfg["prompts"]["dir"]).resolve())
    if "eval" in cfg:
        for key in ("golden", "report"):
            cfg["eval"][key] = str((ROOT / cfg["eval"][key]).resolve())
    return cfg
