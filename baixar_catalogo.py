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
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import yaml
from PIL import Image


OFF_API    = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
OFF_FIELDS = "code,product_name,brands,image_front_url,images"
OFF_SEARCH = "https://search.openfoodfacts.org/search"   # Search-a-licious (v2)
# Licença padrão das imagens do Open Food Facts
OFF_IMAGE_LICENSE = "Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)"
OFF_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/3.0/"


def url_tamanho_cheio(url: str) -> Optional[str]:
    """
    O OFF serve imagens em vários tamanhos: front_pt.11.400.jpg (400 px) e
    front_pt.11.full.jpg (resolução original).  Transforma a URL substituindo
    o sufixo numérico de tamanho (.<N>.jpg) por .full.jpg.
    Retorna None se o padrão não for encontrado (URL já é full ou outro formato).
    """
    novo = re.sub(r"\.\d+\.jpg$", ".full.jpg", url, flags=re.IGNORECASE)
    return novo if novo != url else None


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


def download_image_maior(
    url: str, session: requests.Session, logger: logging.Logger
) -> tuple:
    """
    Tenta baixar a variante em tamanho cheio (.<N>.jpg → .full.jpg) antes da
    versão redimensionada.  Cai para a URL original apenas em 404 ou erro de rede.
    Retorna (bytes_ou_None, variante_str) onde variante_str é "full", "400px" ou "".
    """
    full_url = url_tamanho_cheio(url)
    if full_url:
        try:
            resp = session.get(full_url, timeout=20)
            if resp.status_code == 200:
                logger.info("    variante: full  (%d KB)", len(resp.content) // 1024)
                return resp.content, "full"
            if resp.status_code != 404:
                logger.warning("    full URL HTTP %d, tentando 400px", resp.status_code)
        except requests.RequestException as exc:
            logger.debug("    full URL falhou (%s), usando 400px", exc)

    raw = download_image(url, session, logger)
    if raw:
        logger.info("    variante: 400px (%d KB)", len(raw) // 1024)
        return raw, "400px"
    return None, ""


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
        licenca = c.get("licenca", OFF_IMAGE_LICENSE)
        row = (
            f"| {c['arquivo']}.jpg "
            f"| {c['barcode']} "
            f"| {c['product_name']} "
            f"| {c['brands']} "
            f"| {licenca} |"
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
        raw, _variante = download_image_maior(image_url, session, logger)
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
            entry = {
                "arquivo": arquivo,
                "barcode": parts[1],
                "product_name": parts[2],
                "brands": parts[3],
            }
            if len(parts) >= 5:
                entry["licenca"] = parts[4]
            credits.append(entry)
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
# Modo lote — utilitários
# ---------------------------------------------------------------------------

def normalizar_nome(text: str, max_len: int = 40) -> str:
    """Texto livre → nome de arquivo: minúsculo, sem acento, sem espaço."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[\s\-/\\]+", "_", text)
    text = re.sub(r"[^\w]", "", text)   # mantém apenas [a-z0-9_]
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text[:max_len].rstrip("_")


def gerar_nome_unico(nome_base: str, nomes_usados: set) -> str:
    """Garante unicidade adicionando _2, _3 … em caso de colisão."""
    if nome_base not in nomes_usados:
        return nome_base
    n = 2
    while True:
        sufixo = f"_{n}"
        candidato = nome_base[:40 - len(sufixo)] + sufixo
        if candidato not in nomes_usados:
            return candidato
        n += 1


def buscar_off(
    busca: Optional[str],
    limite: int,
    session: requests.Session,
    logger: logging.Logger,
) -> list[dict]:
    """
    Busca produtos no OFF via Search-a-licious, filtrando por Brasil.
    Retorna lista normalizada com code, product_name, brands, image_front_url.

    Endpoint: https://search.openfoodfacts.org/search
    - q=""  funciona como coringa (sem --busca retorna ~10 000 produtos do Brasil)
    - brands chega como lista; aqui é normalizado para string
    - Retry com backoff exponencial em 503/429
    """
    BACKOFF_BASE = 2.0
    MAX_RETRIES  = 3
    PAGE_SIZE    = 100

    resultados: list[dict] = []
    codigos_vistos: set = set()
    pagina = 1

    while len(resultados) < limite:
        params: dict = {
            "q":              busca if busca else "",
            "countries_tags": "en:brazil",
            "page_size":      PAGE_SIZE,
            "page":           pagina,
            "fields":         "code,product_name,brands,image_front_url",
        }

        logger.info("  [search-a-licious] página %d  (coletados=%d / limite=%d)...",
                    pagina, len(resultados), limite)

        resp = None
        for tentativa in range(1, MAX_RETRIES + 1):
            try:
                resp = session.get(OFF_SEARCH, params=params, timeout=20)
            except requests.RequestException as exc:
                logger.warning("  Erro de rede (tentativa %d/%d): %s",
                               tentativa, MAX_RETRIES, exc)
                if tentativa < MAX_RETRIES:
                    time.sleep(BACKOFF_BASE ** tentativa)
                resp = None
                continue

            if resp.status_code == 200:
                break
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", BACKOFF_BASE ** tentativa))
                logger.warning("  429 Rate-limit — aguardando %ds...", wait)
                time.sleep(wait)
            elif resp.status_code == 503:
                wait = BACKOFF_BASE ** tentativa
                logger.warning("  503 (tentativa %d/%d) — aguardando %.0fs...",
                               tentativa, MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                logger.warning("  HTTP %d inesperado, abortando busca.", resp.status_code)
                break

        if resp is None or resp.status_code != 200:
            logger.warning("  Busca abortada após %d tentativas.", MAX_RETRIES)
            break

        try:
            data = resp.json()
        except ValueError as exc:
            logger.warning("  Resposta não é JSON válido: %s", exc)
            break

        hits = data.get("hits", [])
        if not hits:
            logger.info("  Sem mais resultados.")
            break

        for p in hits:
            if len(resultados) >= limite:
                break
            code      = str(p.get("code") or "").strip()
            pname     = (p.get("product_name") or "").strip()
            image_url = (p.get("image_front_url") or "").strip()
            if not code or not pname or not image_url:
                continue
            if code in codigos_vistos:
                continue

            # Normaliza brands: Search-a-licious devolve lista, não string
            brands_raw = p.get("brands")
            if isinstance(brands_raw, list):
                brands = ", ".join(str(b) for b in brands_raw if b)
            else:
                brands = str(brands_raw or "").strip()

            codigos_vistos.add(code)
            resultados.append({
                "code":            code,
                "product_name":    pname,
                "brands":          brands,
                "image_front_url": image_url,
            })

        # Página incompleta → última página disponível
        if len(hits) < PAGE_SIZE:
            logger.info("  Última página (retornou %d < %d).", len(hits), PAGE_SIZE)
            break

        pagina += 1
        time.sleep(1.0)   # respeita o servidor entre páginas

    logger.info("  Busca concluída: %d produto(s) com dados completos.", len(resultados))
    return resultados


def importar_manual(
    catalog_dir: Path,
    manual_dir: Path,
    logger: logging.Logger,
) -> list[dict]:
    """
    Copia catalogo_manual/*.jpg para catalogo/ quando ainda não há equivalente.
    Retorna lista de dicts de crédito para acrescentar ao CREDITOS.md.
    """
    if not manual_dir.is_dir():
        return []

    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    copiados: list[dict] = []
    for src in sorted(manual_dir.iterdir()):
        if src.suffix.lower() not in EXTS:
            continue
        dest = catalog_dir / f"{src.stem}.jpg"
        if dest.exists():
            logger.debug("  [manual] %s: já existe em catalogo/, ignorado", src.stem)
            continue
        try:
            pil = Image.open(src).convert("RGB")
            pil.save(dest, "JPEG", quality=95)
        except Exception as exc:
            logger.warning("  [manual] %s: falha ao copiar (%s)", src.stem, exc)
            continue
        logger.info("  [manual] %s → catalogo/", src.name)
        copiados.append({
            "arquivo":      src.stem,
            "barcode":      "—",
            "product_name": src.stem.replace("_", " "),
            "brands":       "—",
            "licenca":      "Manual (fornecida pelo usuário)",
        })
    return copiados


def atualizar_csv(csv_path: Path, novos_items: list[dict]):
    """
    Acrescenta itens ao produtos.csv (cria com cabeçalho se não existir).
    Cada item: {"barcode": str, "filename": str}.
    Não duplica linhas já presentes (compara por codigo_de_barras).
    """
    codigos_existentes: set = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                codigos_existentes.add(row.get("codigo_de_barras", "").strip())

    novos = [i for i in novos_items if i["barcode"] not in codigos_existentes]
    if not novos:
        return

    precisa_cabecalho = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["codigo_de_barras", "nome_do_arquivo"])
        if precisa_cabecalho:
            writer.writeheader()
        for item in novos:
            writer.writerow({
                "codigo_de_barras": item["barcode"],
                "nome_do_arquivo":  item["filename"],
            })


# ---------------------------------------------------------------------------
# Pipeline de importação em lote
# ---------------------------------------------------------------------------

def process_lote(
    busca: Optional[str],
    limite: int,
    csv_path: Path,
    config: dict,
    logger: logging.Logger,
):
    dl           = config["download"]
    catalog_dir  = Path(dl["catalog_dir"])
    rejected_dir = Path(dl["rejected_dir"])
    manual_dir   = Path(config.get("catalog_manual_dir", "catalogo_manual"))
    catalog_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    min_w, min_h  = dl["min_width"], dl["min_height"]
    min_sharpness = dl["min_sharpness"]
    delay         = dl["request_delay"]

    session = requests.Session()
    session.headers["User-Agent"] = dl["user_agent"]

    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Nomes já em disco (evita colisão ao gerar novos nomes)
    nomes_usados: set = {
        p.stem for p in catalog_dir.iterdir() if p.suffix.lower() in EXTS
    }
    # Códigos já registrados no CSV (evita rebaixar)
    codigos_no_csv: set = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                codigos_no_csv.add(row.get("codigo_de_barras", "").strip())
        logger.info("CSV existente: %d produto(s) já registrado(s).", len(codigos_no_csv))

    # Busca
    logger.info("Iniciando busca no Open Food Facts  (busca=%r, limite=%d)...",
                busca or "<todos>", limite)
    produtos = buscar_off(busca, limite, session, logger)

    stats: dict = {"ok": 0, "ja_existe": 0, "rejeitado": 0, "erro": 0, "manual": 0}
    motivos: dict = {}
    credits_novos: list[dict] = []
    csv_novos:     list[dict] = []

    for i, prod in enumerate(produtos, 1):
        code      = prod["code"]
        pname     = prod["product_name"]
        brands    = prod["brands"]
        image_url = prod["image_front_url"]

        logger.info("[%d/%d] %s | %s | %s", i, len(produtos), code, brands, pname)

        if code in codigos_no_csv:
            logger.info("  já no CSV, pulando.")
            stats["ja_existe"] += 1
            time.sleep(delay)
            continue

        # Gerar nome único
        prefixo   = f"{brands}_{pname}" if brands else pname
        nome_base = normalizar_nome(prefixo) or normalizar_nome(pname) or code
        filename  = gerar_nome_unico(nome_base, nomes_usados)

        dest          = catalog_dir  / f"{filename}.jpg"
        rejected_dest = rejected_dir / f"{filename}.jpg"

        if dest.exists():
            logger.info("  arquivo já existe, registrando no CSV.")
            nomes_usados.add(filename)
            codigos_no_csv.add(code)
            csv_novos.append({"barcode": code, "filename": filename})
            stats["ja_existe"] += 1
            time.sleep(delay)
            continue

        raw, _variante = download_image_maior(image_url, session, logger)
        if raw is None:
            stats["erro"] += 1
            time.sleep(delay)
            continue

        accepted, reason = check_image(raw, min_w, min_h, min_sharpness)
        if not accepted:
            logger.warning("  REJEITADO (%s): %s", filename, reason)
            save_image(raw, rejected_dest)
            (rejected_dir / f"{filename}.txt").write_text(reason + "\n", encoding="utf-8")
            if "resolução" in reason or "resolucao" in reason:
                chave = "resolução baixa"
            elif "foco" in reason or "laplaciano" in reason:
                chave = "imagem fora de foco"
            else:
                chave = "outro"
            motivos[chave] = motivos.get(chave, 0) + 1
            stats["rejeitado"] += 1
        else:
            save_image(raw, dest)
            nomes_usados.add(filename)
            codigos_no_csv.add(code)
            logger.info("  OK  %s", filename)
            credits_novos.append({
                "arquivo":      filename,
                "barcode":      code,
                "product_name": pname,
                "brands":       brands,
            })
            csv_novos.append({"barcode": code, "filename": filename})
            stats["ok"] += 1

        time.sleep(delay)

    # Importar imagens da pasta manual
    credits_manuais = importar_manual(catalog_dir, manual_dir, logger)
    stats["manual"] = len(credits_manuais)

    # Persistir CSV
    if csv_novos:
        atualizar_csv(csv_path, csv_novos)
        logger.info("produtos.csv atualizado (+%d linha(s)): %s",
                    len(csv_novos), csv_path)

    # Atualizar CREDITOS.md
    todos = credits_novos + credits_manuais
    if todos:
        merged = _merge_credits(_read_existing_credits(catalog_dir), todos)
        write_credits(merged, catalog_dir)

    # Resumo
    logger.info("")
    logger.info("=== Resumo (lote) ===")
    logger.info("  Aceitos (API):       %d", stats["ok"])
    logger.info("  Já existiam:         %d", stats["ja_existe"])
    logger.info("  Rejeitados:          %d", stats["rejeitado"])
    for motivo, n in sorted(motivos.items(), key=lambda x: -x[1]):
        logger.info("    %-28s %d", motivo + ":", n)
    logger.info("  Erros de download:   %d", stats["erro"])
    logger.info("  Do catalogo_manual/: %d", stats["manual"])
    logger.info("  CREDITOS.md:         %s", catalog_dir / "CREDITOS.md")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baixa imagens do Open Food Facts para catalogo/."
    )
    parser.add_argument("--csv", default="produtos.csv",
                        help="CSV de entrada (modo normal) ou saída (--lote)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--lote", action="store_true",
                        help="Importação em lote sem código de barras")
    parser.add_argument("--busca", default=None, metavar="TERMO",
                        help="Termo de busca para --lote (ex: 'iogurte')")
    parser.add_argument("--limite", type=int, default=30,
                        help="Máximo de produtos a importar em --lote (padrão: 30)")
    args = parser.parse_args()

    config   = load_config(args.config)
    logger   = setup_logging(Path("logs"))
    csv_path = Path(args.csv)

    if args.lote:
        process_lote(args.busca, args.limite, csv_path, config, logger)
    else:
        if not csv_path.exists():
            sys.exit(f"Arquivo não encontrado: {csv_path}")
        process(csv_path, config, logger)


if __name__ == "__main__":
    main()
