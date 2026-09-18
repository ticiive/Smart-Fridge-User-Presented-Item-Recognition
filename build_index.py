"""
Lê a pasta catalogo/, calcula o embedding CLIP de cada imagem e salva o
índice em disco (config.yaml → index.path).

Uso:
    python build_index.py
    python build_index.py --catalogo outra_pasta/ --config outro_config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import yaml
from PIL import Image


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_index(catalog_dir: Path, config: dict, device: str):
    model_name = config["recognition"]["clip_model"]
    pretrained = config["recognition"]["clip_pretrained"]

    print(f"Dispositivo: {device}")
    print(f"Modelo: {model_name} / {pretrained}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()

    images = sorted(
        p for p in catalog_dir.iterdir() if p.suffix.lower() in SUPPORTED
    )
    if not images:
        sys.exit(f"Nenhuma imagem encontrada em {catalog_dir}")

    names = []
    embeddings = []

    for img_path in images:
        name = img_path.stem
        print(f"  indexando: {name}")
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        names.append(name)
        embeddings.append(emb.cpu().float().numpy()[0])

    index_path = Path(config["index"]["path"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(index_path, names=np.array(names), embeddings=np.array(embeddings))
    print(f"\nÍndice salvo em {index_path} ({len(names)} produto(s))")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogo", default="catalogo/")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    catalog_dir = Path(args.catalogo)
    if not catalog_dir.is_dir():
        sys.exit(f"Pasta não encontrada: {catalog_dir}")

    device = get_device()
    build_index(catalog_dir, config, device)


if __name__ == "__main__":
    main()
