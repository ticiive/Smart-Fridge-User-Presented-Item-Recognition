"""
Identifica um produto comparando seu embedding CLIP com o catálogo.

Uso:
    python identify.py foto.jpg
    python identify.py foto.jpg --config config.yaml --index index/embeddings.npz
    python identify.py foto.jpg --threshold 0.8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from recognizer import embed_image, get_device, load_index, load_model, match


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagem", help="Caminho da imagem de teste")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--index", default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Substitui o limiar do config.yaml para este run")
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

    threshold = args.threshold if args.threshold is not None \
        else config["recognition"]["similarity_threshold"]

    device = get_device()
    model, preprocess = load_model(config, device)
    names, embeddings = load_index(index_path)

    print(f"Dispositivo: {device}  |  Catálogo: {len(names)} produto(s)  |  Limiar: {threshold}")

    query = embed_image(img_path, model, preprocess, device)
    prediction, top_sim, margin = match(query, names, embeddings, threshold)

    print(f"\nImagem: {img_path}")
    if prediction is None:
        print(f"Resultado: NAO RECONHECIDO  (melhor sim: {top_sim:.4f} < {threshold})")
    else:
        print(f"Resultado:    {prediction}")
        print(f"Similaridade: {top_sim:.4f}  |  Margem: {margin:.4f}")

    sims = embeddings @ query
    print("\nTop 3:")
    for rank, idx in enumerate(np.argsort(sims)[::-1][:3], 1):
        marker = " <--" if names[idx] == prediction else ""
        print(f"  {rank}. {names[idx]:30s}  {sims[idx]:.4f}{marker}")


if __name__ == "__main__":
    main()
