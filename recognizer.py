"""
Núcleo de reconhecimento — carrega modelo CLIP, calcula embeddings e faz
matching por similaridade de cosseno. Importado por identify.py, avaliar.py
e build_index.py para evitar duplicação.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
from PIL import Image


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(config: dict, device: str):
    """Carrega modelo CLIP. Retorna (model, preprocess)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        config["recognition"]["clip_model"],
        pretrained=config["recognition"]["clip_pretrained"],
        device=device,
    )
    model.eval()
    return model, preprocess


def load_index(index_path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(index_path, allow_pickle=True)
    return data["names"].tolist(), data["embeddings"].astype(np.float32)


def embed_image(img_path: Path, model, preprocess, device: str) -> np.ndarray:
    """Retorna embedding L2-normalizado da imagem (shape: [D])."""
    img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()[0]


def match(
    query: np.ndarray,
    names: list[str],
    embeddings: np.ndarray,
    threshold: float,
) -> tuple[Optional[str], float, float]:
    """
    Compara query com o catálogo.
    Retorna (predicao, melhor_similaridade, margem_sobre_segundo).
    predicao é None quando melhor_similaridade < threshold.
    """
    sims = embeddings @ query  # dot product de vetores normalizados = cosine sim
    sorted_idx = np.argsort(sims)[::-1]
    top_sim = float(sims[sorted_idx[0]])
    second_sim = float(sims[sorted_idx[1]]) if len(sims) >= 2 else 0.0
    margin = top_sim - second_sim
    if top_sim < threshold:
        return None, top_sim, margin
    return names[int(sorted_idx[0])], top_sim, margin
