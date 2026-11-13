from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def write_bundle(
    output_dir: str | Path,
    task: str,
    model_version: str,
    model,
    vectorizer,
    metadata: dict,
) -> Path:
    bundle_dir = Path(output_dir) / task / model_version
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, bundle_dir / "model.joblib")
    joblib.dump(vectorizer, bundle_dir / "vectorizer.joblib")

    metadata = {
        **metadata,
        "task": task,
        "version": model_version,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit-learn": _package_version("scikit-learn"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "imbalanced-learn": _package_version("imbalanced-learn"),
        },
    }
    (bundle_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return bundle_dir


def write_manifest(output_dir: str | Path, versions: dict[str, str]) -> Path:
    path = Path(output_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": versions,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
