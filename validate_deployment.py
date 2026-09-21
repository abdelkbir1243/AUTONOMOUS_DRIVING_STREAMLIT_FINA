"""Validation rapide du dépôt avant envoi sur GitHub."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from checkpoint_files import checkpoint_parts, resolve_checkpoint
from model import load_model


def main() -> None:
    root = Path(__file__).resolve().parent
    checkpoint_source = root / "models" / "best_detector.pt"
    checkpoint = resolve_checkpoint(checkpoint_source)
    required = [root / name for name in ("app.py", "model.py", "requirements.txt")]
    missing = [str(path.name) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Fichiers manquants : {', '.join(missing)}")
    if not checkpoint.is_file():
        raise RuntimeError("best_detector.pt ou ses morceaux .partNNN sont absents.")

    with checkpoint.open("rb") as checkpoint_file:
        prefix = checkpoint_file.read(80)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError("Le checkpoint est un pointeur Git LFS, pas le fichier réel.")

    size_mb = checkpoint.stat().st_size / (1024 * 1024)
    parts = checkpoint_parts(checkpoint_source)
    if parts:
        print(f"Morceaux GitHub : {len(parts)}")
    print(f"Checkpoint : {size_mb:.1f} MiB")
    model = load_model(checkpoint, torch.device("cpu"))
    print(f"Architecture chargée : {type(model.backbone).__name__}")
    print("VALIDATION RÉUSSIE — dépôt prêt pour Streamlit.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION ÉCHOUÉE — {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
