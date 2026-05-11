from __future__ import annotations

from pathlib import Path


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()

