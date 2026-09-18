"""
Recebe o caminho de uma imagem de teste e identifica o produto mais parecido
usando similaridade de cosseno contra o índice CLIP.

Uso:
    python identify.py caminho/para/foto.jpg
    python identify.py caminho/para/foto.jpg --config config.yaml --index index/embeddings.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import yaml
from PIL import Image


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_index(index_path: Path):
    data = np.load(index_path, allow_pickle=True)
    return data["names"].tolist(), data["embeddings"]  # list[str], ndarray (N, D)


def embed_image(img_path: Path, model, preprocess, device: str) -> np.ndarray:
    img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()[0]


def identify(img_path: Path, config: dict, index_path: Path):
    threshold = config["recognition"]["similarity_threshold"]
    model_name = config["recognition"]["clip_model"]
    pretrained = config["recognition"]["clip_pretrained"]

    device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()

    names, catalog_embs = load_index(index_path)
    catalog_embs = catalog_embs.astype(np.float32)

    query = embed_image(img_path, model, preprocess, device)

    sims = catalog_embs @ query  # cosine similarity (vetores já normalizados)

    top_idx = int(np.argmax(sims))
    top_sim = float(sims[top_idx])
    top_name = names[top_idx]

    second_sim = float(np.partition(sims, -2)[-2]) if len(sims) >= 2 else 0.0
    margin = top_sim - second_sim

    print(f"\nImagem: {img_path}")
    print(f"Dispositivo: {device}")
    print(f"Limiar de reconhecimento: {threshold}")
    print()

    if top_sim < threshold:
        print(f"Resultado: NAO RECONHECIDO (melhor similaridade: {top_sim:.4f} < {threshold})")
        return None, top_sim

    print(f"Resultado: {top_name}")
    print(f"Similaridade: {top_sim:.4f}  |  Margem sobre 2º: {margin:.4f}")

    print("\nTop 3:")
    top3_idx = np.argsort(sims)[::-1][:3]
    for rank, idx in enumerate(top3_idx, 1):
        print(f"  {rank}. {names[idx]:30s}  {sims[idx]:.4f}")

    return top_name, top_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagem", help="Caminho da imagem de teste")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--index", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    index_path = Path(args.index) if args.index else Path(config["index"]["path"])

    if not index_path.exists():
        sys.exit(
            f"Índice não encontrado: {index_path}\n"
            "Execute primeiro: python build_index.py"
        )

    img_path = Path(args.imagem)
    if not img_path.exists():
        sys.exit(f"Imagem não encontrada: {img_path}")

    identify(img_path, config, index_path)


if __name__ == "__main__":
    main()
