"""
Monta a pasta catalogo/ baixando imagens do Open Food Facts.

Entrada: produtos.csv com colunas  codigo_de_barras, nome_do_arquivo
  - nome_do_arquivo: sem extensão, sem acento, sem espaço
    (será usado como nome do arquivo em catalogo/)

Saída:
  catalogo/<nome_do_arquivo>.jpg   — imagens aceitas
  catalogo_rejeitado/<nome>.jpg    — imagens rejeitadas (com motivo no log)
  catalogo/CREDITOS.md             — atribuição obrigatória para repo público
  logs/download_<timestamp>.log    — log completo

Uso:
  python baixar_catalogo.py
  python baixar_catalogo.py --csv outro.csv --config outro_config.yaml
"""

import argparse
import csv
import io
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import yaml
from PIL import Image


OFF_API = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
OFF_FIELDS = "code,product_name,brands,image_front_url,images"
# Licença padrão das imagens do Open Food Facts
OFF_IMAGE_LICENSE = "Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)"
OFF_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/3.0/"


# ---------------------------------------------------------------------------
# Configuração e logging
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"download_{timestamp}.log"

    logger = logging.getLogger("baixar_catalogo")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log em: %s", log_file)
    return logger


# ---------------------------------------------------------------------------
# Consulta à API
# ---------------------------------------------------------------------------

def query_off(barcode: str, session: requests.Session, logger: logging.Logger
              ) -> Optional[dict]:
    url = OFF_API.format(barcode=barcode)
    try:
        resp = session.get(url, params={"fields": OFF_FIELDS}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("[%s] Falha na requisição: %s", barcode, exc)
        return None

    data = resp.json()
    if data.get("status") != 1:
        logger.warning("[%s] Produto não encontrado no Open Food Facts.", barcode)
        return None

    return data.get("product", {})


# ---------------------------------------------------------------------------
# Download e validação da imagem
# ---------------------------------------------------------------------------

def download_image(url: str, session: requests.Session,
                   logger: logging.Logger) -> Optional[bytes]:
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        logger.warning("Erro ao baixar imagem (%s): %s", url, exc)
        return None


def check_image(raw: bytes, min_w: int, min_h: int,
                min_sharpness: float) -> tuple[bool, str]:
    """Retorna (aceito, motivo_rejeição). motivo vazio = aceito."""
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        return False, f"não foi possível decodificar a imagem: {exc}"

    w, h = pil.size
    if w < min_w or h < min_h:
        return False, f"resolução {w}x{h} abaixo do mínimo {min_w}x{min_h}"

    # Nitidez via variância do Laplaciano
    gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < min_sharpness:
        return False, (
            f"imagem fora de foco (laplaciano={sharpness:.1f} < {min_sharpness})"
        )

    return True, ""


def save_image(raw: bytes, dest: Path):
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    pil.save(dest, "JPEG", quality=95)


# ---------------------------------------------------------------------------
# Créditos
# ---------------------------------------------------------------------------

def write_credits(credits: list[dict], catalog_dir: Path):
    lines = [
        "# Créditos das imagens do catálogo",
        "",
        "Imagens obtidas do [Open Food Facts](https://world.openfoodfacts.org/),",
        f"licenciadas sob [{OFF_IMAGE_LICENSE}]({OFF_LICENSE_URL}).",
        "O banco de dados Open Food Facts é disponibilizado sob a",
        "[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1.0/).",
        "",
        "| Arquivo | Código de barras | Nome do produto | Marca | Licença |",
        "|---------|-----------------|-----------------|-------|---------|",
    ]
    for c in credits:
        row = (
            f"| {c['arquivo']}.jpg "
            f"| {c['barcode']} "
            f"| {c['product_name']} "
            f"| {c['brands']} "
            f"| {OFF_IMAGE_LICENSE} |"
        )
        lines.append(row)

    credits_path = catalog_dir / "CREDITOS.md"
    credits_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def process(csv_path: Path, config: dict, logger: logging.Logger):
    dl = config["download"]
    catalog_dir = Path(dl["catalog_dir"])
    rejected_dir = Path(dl["rejected_dir"])
    catalog_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    min_w = dl["min_width"]
    min_h = dl["min_height"]
    min_sharpness = dl["min_sharpness"]
    delay = dl["request_delay"]

    session = requests.Session()
    session.headers["User-Agent"] = dl["user_agent"]

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"codigo_de_barras", "nome_do_arquivo"}
        if not expected.issubset(set(reader.fieldnames or [])):
            sys.exit(
                f"O CSV precisa ter as colunas: {expected}\n"
                f"Colunas encontradas: {reader.fieldnames}"
            )
        rows = list(reader)

    logger.info("Produtos no CSV: %d", len(rows))

    stats = {"ok": 0, "sem_produto": 0, "sem_imagem": 0, "rejeitado": 0, "erro": 0}
    credits: list[dict] = []

    for i, row in enumerate(rows, 1):
        barcode = row["codigo_de_barras"].strip()
        filename = row["nome_do_arquivo"].strip()
        dest = catalog_dir / f"{filename}.jpg"
        rejected_dest = rejected_dir / f"{filename}.jpg"

        logger.info("[%d/%d] %s → %s", i, len(rows), barcode, filename)

        # Não reprocessar se já existir
        if dest.exists():
            logger.info("  já existe, pulando.")
            stats["ok"] += 1
            continue

        product = query_off(barcode, session, logger)
        if product is None:
            stats["sem_produto"] += 1
            time.sleep(delay)
            continue

        image_url = product.get("image_front_url")
        if not image_url:
            logger.warning("  [%s] Sem imagem frontal na base.", barcode)
            stats["sem_imagem"] += 1
            time.sleep(delay)
            continue

        logger.debug("  imagem: %s", image_url)
        raw = download_image(image_url, session, logger)
        if raw is None:
            stats["erro"] += 1
            time.sleep(delay)
            continue

        accepted, reason = check_image(raw, min_w, min_h, min_sharpness)
        if not accepted:
            logger.warning("  REJEITADO (%s): %s", filename, reason)
            save_image(raw, rejected_dest)
            # Registra motivo em txt junto do arquivo rejeitado
            (rejected_dir / f"{filename}.txt").write_text(
                reason + "\n", encoding="utf-8"
            )
            stats["rejeitado"] += 1
        else:
            save_image(raw, dest)
            product_name = product.get("product_name") or ""
            brands = product.get("brands") or ""
            logger.info("  OK  %s | %s | %s", filename, product_name, brands)
            credits.append({
                "arquivo": filename,
                "barcode": barcode,
                "product_name": product_name,
                "brands": brands,
            })
            stats["ok"] += 1

        time.sleep(delay)

    # Atualiza créditos somente com as imagens aceitas desta execução
    # (re-lê entradas já existentes para não sobrescrever corridas anteriores)
    existing_credits = _read_existing_credits(catalog_dir)
    merged = _merge_credits(existing_credits, credits)
    if merged:
        write_credits(merged, catalog_dir)

    logger.info("")
    logger.info("=== Resumo ===")
    logger.info("  Aceitos:       %d", stats["ok"])
    logger.info("  Rejeitados:    %d", stats["rejeitado"])
    logger.info("  Sem produto:   %d", stats["sem_produto"])
    logger.info("  Sem imagem:    %d", stats["sem_imagem"])
    logger.info("  Erros HTTP:    %d", stats["erro"])
    logger.info("  CREDITOS.md:   %s", catalog_dir / "CREDITOS.md")


def _read_existing_credits(catalog_dir: Path) -> list[dict]:
    """Lê créditos já registrados numa corrida anterior (parse simples da tabela MD)."""
    cred_path = catalog_dir / "CREDITOS.md"
    if not cred_path.exists():
        return []
    credits = []
    for line in cred_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| Arquivo"):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 4:
            arquivo = parts[0].removesuffix(".jpg")
            credits.append({
                "arquivo": arquivo,
                "barcode": parts[1],
                "product_name": parts[2],
                "brands": parts[3],
            })
    return credits


def _merge_credits(existing: list[dict], new: list[dict]) -> list[dict]:
    seen = {c["arquivo"] for c in existing}
    merged = list(existing)
    for c in new:
        if c["arquivo"] not in seen:
            merged.append(c)
            seen.add(c["arquivo"])
    return merged


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="produtos.csv")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(Path("logs"))

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"Arquivo não encontrado: {csv_path}")

    process(csv_path, config, logger)


if __name__ == "__main__":
    main()
