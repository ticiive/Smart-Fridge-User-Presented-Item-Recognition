"""
Lê catalogo/ e catalogo_manual/ (fallback), calcula embeddings CLIP e salva
o índice em disco (config.yaml → index.path).

catalogo/ tem prioridade: se o mesmo nome existir nas duas pastas, a imagem
de catalogo/ (vinda do Open Food Facts) é usada.
Imagens em catalogo_manual/ só entram quando não há versão aceita em catalogo/.

Uso:
    python build_index.py
    python build_index.py --catalogo outra/ --manual outra_manual/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from recognizer import embed_image, get_device, load_model


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def index_dir(
    img_dir: Path,
    model,
    preprocess,
    device: str,
    skip: set[str],
    label: str,
) -> tuple[list[str], list[np.ndarray]]:
    """Indexa imagens em img_dir, pulando nomes já presentes em skip."""
    names, embs = [], []
    for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in SUPPORTED):
        name = img_path.stem
        if name in skip:
            print(f"  [{label}] {name}: já no catálogo principal, ignorado")
            continue
        print(f"  [{label}] {name}")
        embs.append(embed_image(img_path, model, preprocess, device))
        names.append(name)
    return names, embs


def update_credits_manual(catalog_dir: Path, manual_names: list[str]):
    """Acrescenta entradas manuais ao CREDITOS.md sem duplicar linhas."""
    credits_path = catalog_dir / "CREDITOS.md"
    if credits_path.exists():
        existing = credits_path.read_text(encoding="utf-8")
    else:
        existing = (
            "# Créditos das imagens do catálogo\n\n"
            "| Arquivo | Código de barras | Nome do produto | Marca | Licença |\n"
            "|---------|-----------------|-----------------|-------|---------|"
        )
    new_rows = [
        f"| {name}.jpg | — | {name.replace('_', ' ')} | — "
        f"| Manual (fornecida pelo usuário) |"
        for name in manual_names
        if f"| {name}.jpg " not in existing
    ]
    if not new_rows:
        return
    credits_path.write_text(
        existing.rstrip("\n") + "\n" + "\n".join(new_rows) + "\n",
        encoding="utf-8",
    )
    print(f"  CREDITOS.md: +{len(new_rows)} entrada(s) manual(is)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogo", default=None)
    parser.add_argument("--manual", default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    catalog_dir = Path(args.catalogo or config["download"]["catalog_dir"])
    manual_dir = Path(args.manual or config.get("catalog_manual_dir", "catalogo_manual"))
    index_path = Path(config["index"]["path"])

    device = get_device()
    model, preprocess = load_model(config, device)
    print(f"Dispositivo: {device}  |  Modelo: {config['recognition']['clip_model']}")

    cat_names, cat_embs = [], []
    if catalog_dir.is_dir():
        cat_names, cat_embs = index_dir(
            catalog_dir, model, preprocess, device, skip=set(), label="catalogo"
        )
    else:
        print(f"  {catalog_dir} não encontrada, verificando apenas manual.")

    man_names, man_embs = [], []
    if manual_dir.is_dir():
        man_names, man_embs = index_dir(
            manual_dir, model, preprocess, device, skip=set(cat_names), label="manual"
        )

    all_names = cat_names + man_names
    all_embs = cat_embs + man_embs

    if not all_names:
        sys.exit("Nenhuma imagem encontrada para indexar.")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(index_path, names=np.array(all_names), embeddings=np.array(all_embs))

    print(f"\nÍndice → {index_path}")
    print(f"  catalogo/:        {len(cat_names):3d} produto(s)")
    print(f"  catalogo_manual/: {len(man_names):3d} produto(s) (fallback)")
    print(f"  total:            {len(all_names):3d}")

    if man_names and catalog_dir.is_dir():
        update_credits_manual(catalog_dir, man_names)


if __name__ == "__main__":
    main()
