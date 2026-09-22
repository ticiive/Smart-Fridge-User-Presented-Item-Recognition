"""
processar_capturas.py — pipeline offline de tres opinioes por recorte.

Processa imagens de logs_demo/<sessao>/ e capturas/ (se existir),
emitindo um CSV com tres opinioes independentes por recorte:

  1. clip_b32  — embedding CLIP ViT-B-32, top-1 vs. catalogo
  2. ocr       — Apple Vision OCR (VNRecognizeTextRequest, nivel accurate,
                 pt-BR + en-US) com match por palavras-chave em
                 catalogo_palavras.yaml
  3. vlm       — Ollama qwen2.5vl:3b, lista do catalogo enviada no prompt,
                 temperatura 0

Uso:
  python processar_capturas.py
  python processar_capturas.py --sessao logs_demo/20260921_143000
  python processar_capturas.py --catalogo outra_pasta/
  python processar_capturas.py --gerar-palavras-yaml
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
import yaml
from PIL import Image


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MODELO_CLIP    = "ViT-B-32"
PRETRAINED     = "laion2b_s34b_b79k"
EXTENSOES      = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PALAVRAS_YAML  = Path("catalogo_palavras.yaml")
OLLAMA_URL     = "http://localhost:11434"
OLLAMA_MODELO  = "qwen2.5vl:3b"
OLLAMA_TIMEOUT = 30   # segundos por requisicao
RESULTADOS_DIR = Path("resultados")
CAPTURAS_DIR   = Path("capturas")
LOGS_DEMO_DIR  = Path("logs_demo")


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Catalogo
# ---------------------------------------------------------------------------

def carregar_catalogo(
    catalog_dir: Path,
    model,
    preprocess,
    device: str,
) -> tuple[list[str], np.ndarray]:
    imagens = sorted(
        p for p in catalog_dir.iterdir() if p.suffix.lower() in EXTENSOES
    )
    nomes, embs = [], []
    for img_path in imagens:
        pil    = Image.open(img_path).convert("RGB")
        tensor = preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        nomes.append(img_path.stem)
        embs.append(emb.cpu().float().numpy()[0])
    return nomes, np.array(embs, dtype=np.float32)


def gerar_palavras_yaml(nomes: list[str]) -> dict:
    resultado: dict = {}
    for nome in nomes:
        vistos: set = set()
        palavras = []
        for parte in nome.split("_"):
            pn = normalizar(parte)
            if pn and len(pn) > 1 and not pn.isdigit() and pn not in vistos:
                vistos.add(pn)
                palavras.append(pn)
        resultado[nome] = palavras
    return resultado


# ---------------------------------------------------------------------------
# Opiniao 1 — CLIP B32
# ---------------------------------------------------------------------------

def inferir_clip(
    img_path: Path,
    model,
    preprocess,
    device: str,
    nomes: list[str],
    embeddings: np.ndarray,
) -> tuple[str, float]:
    pil    = Image.open(img_path).convert("RGB")
    tensor = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    query = emb.cpu().float().numpy()[0]
    sims  = embeddings @ query
    idx   = int(np.argmax(sims))
    return nomes[idx], float(sims[idx])


# ---------------------------------------------------------------------------
# Opiniao 2 — OCR Apple Vision
# ---------------------------------------------------------------------------

_vision_ok: Optional[bool] = None


def checar_vision() -> bool:
    global _vision_ok
    if _vision_ok is not None:
        return _vision_ok
    try:
        import Vision          # noqa: F401
        from Foundation import NSURL  # noqa: F401
        _vision_ok = True
    except ImportError:
        print(
            "AVISO: pyobjc-framework-Vision nao instalado — OCR desativado.\n"
            "  Instale com: .venv/bin/pip install pyobjc-framework-Vision\n"
            "  Ou todo o bundle:  .venv/bin/pip install pyobjc"
        )
        _vision_ok = False
    return _vision_ok


def ocr_imagem(img_path: Path) -> str:
    try:
        import Vision
        from Foundation import NSURL

        url     = NSURL.fileURLWithPath_(str(img_path.resolve()))
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_(["pt-BR", "en-US"])
        request.setUsesLanguageCorrection_(True)

        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
        handler.performRequests_error_([request], None)

        results = request.results()
        if not results:
            return ""
        partes = []
        for obs in results:
            tops = obs.topCandidates_(1)
            if tops:
                partes.append(str(tops[0].string()))
        return " ".join(partes)
    except Exception as exc:
        print(f"  [ocr] {img_path.name}: {exc}")
        return ""


def match_palavras(texto: str, palavras: dict) -> str:
    if not texto.strip() or not palavras:
        return "nenhum"
    texto_norm  = normalizar(texto)
    melhor      = "nenhum"
    melhor_cnt  = 0
    for nome, kws in palavras.items():
        cnt = sum(1 for kw in kws if normalizar(kw) in texto_norm)
        if cnt > melhor_cnt:
            melhor_cnt = cnt
            melhor     = nome
    return melhor if melhor_cnt > 0 else "nenhum"


# ---------------------------------------------------------------------------
# Opiniao 3 — VLM Ollama
# ---------------------------------------------------------------------------

_ollama_ok: Optional[bool] = None


def checar_ollama() -> bool:
    global _ollama_ok
    if _ollama_ok is not None:
        return _ollama_ok
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                f"{OLLAMA_URL}/api/tags",
                headers={"User-Agent": "processar_capturas/1.0"},
            ),
            timeout=2,
        )
        _ollama_ok = True
    except Exception:
        print(
            f"AVISO: Ollama nao encontrado em {OLLAMA_URL} — VLM desativado.\n"
            f"  Inicie com: ollama serve\n"
            f"  Modelo necessario: ollama pull {OLLAMA_MODELO}"
        )
        _ollama_ok = False
    return _ollama_ok


def inferir_vlm(img_path: Path, nomes: list[str]) -> str:
    try:
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        lista  = "\n".join(f"- {n}" for n in nomes)
        prompt = (
            "You are analyzing a food product image from a fridge camera.\n"
            "Choose exactly one name from the catalog list below that matches "
            "the product in the image.\n"
            "If no product matches, respond: nenhum\n"
            "Respond with ONLY the exact name from the list or 'nenhum'. "
            "No explanation, no punctuation, nothing else.\n\n"
            f"Catalog:\n{lista}"
        )
        payload = json.dumps({
            "model":   OLLAMA_MODELO,
            "prompt":  prompt,
            "images":  [img_b64],
            "options": {"temperature": 0},
            "stream":  False,
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read())

        resposta   = data.get("response", "").strip().strip("\"'")
        nomes_set  = set(nomes)

        if resposta.lower() == "nenhum":
            return "nenhum"
        if resposta in nomes_set:
            return resposta
        resp_lower = resposta.lower()
        for n in nomes:
            if n.lower() == resp_lower:
                return n
        return f"invalida:{resposta[:60]}"

    except urllib.error.URLError as exc:
        print(f"  [vlm] conexao perdida: {exc}")
        return ""
    except Exception as exc:
        print(f"  [vlm] {img_path.name}: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Coleta de imagens
# ---------------------------------------------------------------------------

def coletar_imagens(sessao: Optional[str]) -> list[Path]:
    imagens: list[Path] = []
    if sessao:
        pasta = Path(sessao)
        if not pasta.is_dir():
            sys.exit(f"Sessao nao encontrada: {pasta}")
        imagens = sorted(
            p for p in pasta.iterdir() if p.suffix.lower() in EXTENSOES
        )
    else:
        if LOGS_DEMO_DIR.is_dir():
            for subdir in sorted(LOGS_DEMO_DIR.iterdir()):
                if subdir.is_dir():
                    imagens += sorted(
                        p for p in subdir.iterdir()
                        if p.suffix.lower() in EXTENSOES
                    )
        if CAPTURAS_DIR.is_dir():
            imagens += sorted(
                p for p in CAPTURAS_DIR.iterdir()
                if p.suffix.lower() in EXTENSOES
            )
    return imagens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline offline: CLIP + OCR (Vision) + VLM (Ollama) por recorte."
    )
    parser.add_argument("--catalogo", default=None,
                        help="Pasta do catalogo (padrao: config.yaml -> catalog_dir)")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--sessao",   default=None,
                        help="Processar apenas uma sessao: logs_demo/<timestamp>")
    parser.add_argument("--gerar-palavras-yaml", action="store_true",
                        help="Gera/sobrescreve catalogo_palavras.yaml e sai")
    args = parser.parse_args()

    config      = load_config(args.config)
    catalog_dir = Path(args.catalogo or config["download"]["catalog_dir"])

    if not catalog_dir.is_dir():
        sys.exit(f"Catalogo nao encontrado: {catalog_dir}")

    # --gerar-palavras-yaml: so precisa dos nomes dos arquivos, nao do modelo
    if args.gerar_palavras_yaml:
        imagens_cat = sorted(
            p for p in catalog_dir.iterdir() if p.suffix.lower() in EXTENSOES
        )
        nomes_cat = [p.stem for p in imagens_cat]
        palavras  = gerar_palavras_yaml(nomes_cat)
        with open(PALAVRAS_YAML, "w", encoding="utf-8") as f:
            f.write(
                "# Palavras-chave por produto para matching de OCR.\n"
                "# Adicione sinonimos, abreviacoes e variacoes de escrita.\n"
                "# Exemplo:  heinz_mostarda: [heinz, mostarda, mustard, ketchup]\n\n"
            )
            yaml.dump(palavras, f, allow_unicode=True,
                      default_flow_style=False, sort_keys=True)
        print(f"Gerado: {PALAVRAS_YAML}  ({len(palavras)} entradas)")
        print("Revise as palavras-chave antes de rodar o pipeline.")
        return

    # Carrega modelo e catalogo
    device = get_device()
    print(f"Dispositivo : {device}")
    print(f"Carregando CLIP ({MODELO_CLIP} / {PRETRAINED})...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        MODELO_CLIP, pretrained=PRETRAINED, device=device
    )
    clip_model.eval()

    print(f"Indexando catalogo em {catalog_dir}...")
    nomes, embeddings = carregar_catalogo(catalog_dir, clip_model, preprocess, device)
    print(f"  {len(nomes)} produto(s) carregado(s).\n")

    # Palavras-chave para OCR
    palavras_ocr: dict = {}
    if PALAVRAS_YAML.exists():
        with open(PALAVRAS_YAML, encoding="utf-8") as f:
            palavras_ocr = yaml.safe_load(f) or {}
        print(f"Palavras OCR: {PALAVRAS_YAML}  ({len(palavras_ocr)} entradas)")
    else:
        print(
            f"AVISO: {PALAVRAS_YAML} nao encontrado — OCR retornara 'nenhum' para tudo.\n"
            "  Gere com: python processar_capturas.py --gerar-palavras-yaml"
        )

    imagens = coletar_imagens(args.sessao)
    if not imagens:
        sys.exit(
            "Nenhuma imagem encontrada em logs_demo/ ou capturas/.\n"
            "Verifique se a demo salvou recortes ou use --sessao."
        )
    print(f"\n{len(imagens)} recorte(s) para processar.\n")

    usar_ocr = checar_vision()
    usar_vlm = checar_ollama()
    print()

    RESULTADOS_DIR.mkdir(exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_out = RESULTADOS_DIR / f"offline_{ts}.csv"

    campos = [
        "arquivo",
        "clip_nome", "clip_sim", "clip_ms",
        "ocr_texto", "ocr_nome", "ocr_ms",
        "vlm_nome", "vlm_ms",
        "verdadeiro",
    ]

    n_total      = 0
    soma_clip_ms = 0.0
    soma_ocr_ms  = 0.0
    soma_vlm_ms  = 0.0

    with open(csv_out, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=campos)
        writer.writeheader()

        for i, img_path in enumerate(imagens, 1):
            print(f"[{i:4d}/{len(imagens)}] {img_path.name}", end="  ", flush=True)

            t0 = time.perf_counter()
            clip_nome, clip_sim = inferir_clip(
                img_path, clip_model, preprocess, device, nomes, embeddings
            )
            clip_ms       = (time.perf_counter() - t0) * 1000
            soma_clip_ms += clip_ms

            t0 = time.perf_counter()
            ocr_texto = ocr_imagem(img_path) if usar_ocr else ""
            ocr_nome  = match_palavras(ocr_texto, palavras_ocr)
            ocr_ms        = (time.perf_counter() - t0) * 1000
            if usar_ocr:
                soma_ocr_ms += ocr_ms

            t0 = time.perf_counter()
            vlm_nome  = inferir_vlm(img_path, nomes) if usar_vlm else ""
            vlm_ms        = (time.perf_counter() - t0) * 1000
            if usar_vlm:
                soma_vlm_ms += vlm_ms

            n_total += 1
            print(
                f"clip={clip_nome[:22]}({clip_sim:.3f})  "
                f"ocr={ocr_nome[:22]}  "
                f"vlm={str(vlm_nome)[:22]}"
            )

            writer.writerow({
                "arquivo":    str(img_path),
                "clip_nome":  clip_nome,
                "clip_sim":   f"{clip_sim:.4f}",
                "clip_ms":    f"{clip_ms:.1f}",
                "ocr_texto":  ocr_texto,
                "ocr_nome":   ocr_nome,
                "ocr_ms":     f"{ocr_ms:.1f}",
                "vlm_nome":   vlm_nome,
                "vlm_ms":     f"{vlm_ms:.1f}",
                "verdadeiro": "",
            })

    print(f"\nResultados: {csv_out}")
    print("\n=== RESUMO ===")
    print(f"  Recortes processados : {n_total}")
    if n_total:
        print(f"  CLIP  : {n_total} classificacoes  |  media {soma_clip_ms / n_total:.1f} ms/img")
        if usar_ocr:
            print(f"  OCR   : {n_total} classificacoes  |  media {soma_ocr_ms / n_total:.1f} ms/img")
        else:
            print("  OCR   : desativado (pyobjc-framework-Vision nao instalado)")
        if usar_vlm:
            print(f"  VLM   : {n_total} classificacoes  |  media {soma_vlm_ms / n_total:.1f} ms/img")
        else:
            print("  VLM   : desativado (Ollama nao encontrado)")


if __name__ == "__main__":
    main()
