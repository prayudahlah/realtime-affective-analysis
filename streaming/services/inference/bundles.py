from __future__ import annotations

import json
from pathlib import Path

import joblib


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_active_version(model_root: str | Path, task: str) -> str:
    root = Path(model_root)
    manifest = root / "manifest.json"
    if manifest.exists():
        versions = _read_json(manifest).get("models", {})
        if task in versions:
            return str(versions[task])
    task_dir = root / task
    candidates = sorted(path.name for path in task_dir.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"Tidak ada bundle untuk task {task} di {task_dir}")
    return candidates[-1]


def load_active_bundle(model_root: str | Path, task: str) -> dict:
    root = Path(model_root)
    version = resolve_active_version(root, task)
    bundle = root / task / version
    metadata = _read_json(bundle / "metadata.json")
    return {
        "version": version,
        "metadata": metadata,
        "model": joblib.load(bundle / "model.joblib"),
        "vectorizer": joblib.load(bundle / "vectorizer.joblib"),
    }
