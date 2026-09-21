"""Crée un checkpoint de déploiement compact à partir du checkpoint Colab."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch


CHUNK_SIZE = 20 * 1024 * 1024


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "detector_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Aucun state_dict de détecteur trouvé dans le checkpoint.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="best_detector.pt produit par Colab")
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=Path("models/best_detector.pt"),
    )
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(args.source)

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    state = extract_state_dict(checkpoint)
    compact = {
        key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
        for key, value in state.items()
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    for old_part in args.destination.parent.glob(f"{args.destination.name}.part*"):
        old_part.unlink()
    torch.save({"model_state_dict": compact, "format": "streamlit-fp16-storage-v1"}, args.destination)

    size_mb = args.destination.stat().st_size / (1024 * 1024)
    print(f"Checkpoint créé : {args.destination}")
    print(f"Taille : {size_mb:.1f} MiB")
    if size_mb >= 100:
        raise RuntimeError("Le checkpoint compact dépasse encore la limite GitHub de 100 MiB.")

    if args.destination.stat().st_size > CHUNK_SIZE:
        with args.destination.open("rb") as source:
            index = 1
            while chunk := source.read(CHUNK_SIZE):
                part = args.destination.with_name(f"{args.destination.name}.part{index:03d}")
                part.write_bytes(chunk)
                print(f"Morceau : {part.name} ({part.stat().st_size / 1024**2:.1f} MiB)")
                index += 1
        args.destination.unlink()
        print("OK : morceaux compatibles avec l'upload web GitHub, sans Git LFS.")
    else:
        print("OK : fichier compatible avec l'upload web GitHub.")


if __name__ == "__main__":
    main()
