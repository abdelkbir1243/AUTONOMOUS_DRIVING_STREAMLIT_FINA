"""Gestion d'un checkpoint unique ou découpé pour l'upload GitHub web."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path


def checkpoint_parts(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}.part*"))


def resolve_checkpoint(path: Path) -> Path:
    """Retourne le checkpoint réel, en reconstituant ses morceaux si nécessaire."""
    if path.is_file():
        return path

    parts = checkpoint_parts(path)
    if not parts:
        return path

    signature = "|".join(
        f"{part.name}:{part.stat().st_size}:{part.stat().st_mtime_ns}"
        for part in parts
    )
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    merged = Path(tempfile.gettempdir()) / f"best_detector_{digest}.pt"
    if merged.is_file() and merged.stat().st_size == sum(p.stat().st_size for p in parts):
        return merged

    temporary = merged.with_suffix(".tmp")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    output.write(chunk)
    temporary.replace(merged)
    return merged
