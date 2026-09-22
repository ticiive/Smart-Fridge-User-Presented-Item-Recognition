"""
Demo ao vivo — apresentacao AP1.

Dois detectores:
  --detector fundo  (padrao) — diferenca contra frame de referencia
  --detector yolo             — YOLO-World vocabulario aberto

Tres modos de registro de inventario:
  --direcao nenhum  (padrao) — reconhecimento automatico por estabilidade
  --direcao area             — direcao por tamanho aparente (trilha)
  --direcao linha            — direcao por cruzamento de linha vertical

Le catalogo/*.jpg diretamente no startup; nao depende de build_index.py.

Uso:
    python demo_ao_vivo.py
    python demo_ao_vivo.py --detector yolo
    python demo_ao_vivo.py --direcao area
    python demo_ao_vivo.py --camera 1
    python demo_ao_vivo.py --listar-cameras
    python demo_ao_vivo.py --catalogo outra_pasta/
    python demo_ao_vivo.py --com-enriquecido

Teclas:
    q        Sair (grava inventario/ e imprime resumo)
    +  /  =  Aumentar limiar em 0.01
    -        Diminuir limiar em 0.01
    b        Capturar fundo de referencia (cena vazia)
    d        Liga/desliga deteccao automatica
    l        Alternar area/linha (so quando --direcao area|linha)
    setas    Mover caixa manualmente (desliga deteccao automatica)
    [  /  ]  Diminuir / aumentar tamanho da caixa
    a        Adicionar produto ao inventario manualmente
    s        Remover produto do inventario manualmente
    r        Zerar inventario e historico de eventos
    c        Salvar recorte em catalogo_enriquecido/
    ESPACO   Salvar recorte em testes/
    <  /  >  Diminuir / aumentar margem minima em 0.005
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
import yaml
from PIL import Image


# ---------------------------------------------------------------------------
# Constantes — ajuste aqui sem mexer no resto
# ---------------------------------------------------------------------------

INFERENCIA_A_CADA_N_FRAMES = 5
LIMIAR_INICIAL             = 0.63
AQUECIMENTO_FRAMES         = 30
LIMIAR_ESCURIDAO           = 5.0

# Deteccao por diferenca de fundo
FUNDO_BLUR_KERNEL = (9, 9)
AREA_MINIMA_FRAC  = 0.03
MARGEM_BOX        = 0.10
SUAVIZACAO_ALPHA  = 0.50

# Detector YOLO-World — edite YOLO_CLASSES para ajustar vocabulario
YOLO_MODELO        = "yolov8s-worldv2.pt"
YOLO_CONF          = 0.05   # confianca baixa para nao perder embalagens pequenas
YOLO_IOU_PESSOA    = 0.50   # IoU maximo entre embalagem e corpo/mao para descartar
YOLO_MAX_AREA_FRAC = 0.25   # fracao maxima do frame — descarta troncos e pessoas inteiras
YOLO_CLASSES = [
    "bag of potato chips",
    "instant ramen packet",
    "package of cookies",
    "soda can",
    "plastic bottle",
    "cardboard food box",
    "person",
    "human face",
    "hand",
    "arm",
]
YOLO_DESCARTAR = {"person", "human face", "hand", "arm"}

# Hortifruti: detectado pelo rotulo do YOLO, sem passar pelo catalogo CLIP
YOLO_CLASSES_FRUTA = [
    "banana", "apple", "orange", "lemon", "tomato", "carrot", "potato",
    "onion", "broccoli", "grapes", "strawberry", "avocado",
]
YOLO_CONF_FRUTA     = 0.50   # limiar de confianca para aceitar fruta/hortifruti
YOLO_IOU_FRUTA_EMB  = 0.30   # IoU maximo entre caixa de fruta e caixa de embalagem
YOLO_MAX_AREA_FRUTA = 0.10   # fruta na mao e pequena; descarta se > 10% do frame
YOLO_FRUTA_PT = {
    "banana":     "banana",
    "apple":      "maca",
    "orange":     "laranja",
    "lemon":      "limao",
    "tomato":     "tomate",
    "carrot":     "cenoura",
    "potato":     "batata",
    "onion":      "cebola",
    "broccoli":   "brocolis",
    "grapes":     "uva",
    "strawberry": "morango",
    "avocado":    "abacate",
}
_YOLO_FRUTA_SET = set(YOLO_CLASSES_FRUTA)  # lookup O(1)

# Modo reconhecimento (--direcao nenhum): decisao ao fim da passagem
PASSAGEM_TIMEOUT        = 8    # frames sem deteccao valida para encerrar a passagem
PASSAGEM_INFERENCIA_N   = 2    # intervalo de inferencia durante a passagem (frames)
MIN_ACERTOS_PASSAGEM    = 1    # minimo de inferencias aceitas para adicionar ao carrinho
PASSAGEM_MIN_FRAMES     = 5    # passagens com menos frames sao ignoradas ao salvar
MARGEM_MINIMA           = 0.02 # diferenca minima entre 1o e 2o do catalogo para aceitar

# VLM local (--vlm)
OLLAMA_URL          = "http://localhost:11434"
OLLAMA_MODELO_VLM   = "qwen2.5vl:3b"
OLLAMA_TIMEOUT_VLM  = 30   # segundos

# Trilha (--direcao area|linha)
TRILHA_FRAMES_TIMEOUT = 10
TRILHA_MIN_FRAMES     = 8
FEEDBACK_DURACAO      = 2.0

# Criterio por area aparente
AREA_JANELA_SUAVE      = 4
AREA_FATOR_ENTRADA     = 1.35
AREA_FATOR_SAIDA       = 0.74
APROXIMANDO_EH_ENTRADA = True

# Criterio por cruzamento de linha
LINHA_X_FRAC = 0.50

# Controle manual da caixa
PASSO_MOVER  = 10
PASSO_RESIZE = 15

KEY_UP    = {63232, 2490368}
KEY_DOWN  = {63233, 2621440}
KEY_LEFT  = {63234, 2424832}
KEY_RIGHT = {63235, 2555904}

MODELO_CLIP = "ViT-B-32"
PRETRAINED  = "laion2b_s34b_b79k"

EXTENSOES_SUPORTADAS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

COR_VERDE    = (80, 200, 80)
COR_VERMELHO = (60, 60, 220)
COR_AMARELO  = (0, 200, 200)
COR_AZUL     = (220, 140, 40)
COR_CINZA    = (170, 170, 170)
COR_BRANCO   = (240, 240, 240)
COR_PRETO    = (0, 0, 0)
ALPHA_FUNDO  = 0.50


# ---------------------------------------------------------------------------
# Device / modelo CLIP
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Catalogo
# ---------------------------------------------------------------------------

def carregar_catalogo(
    catalog_dir: Path, model, preprocess, device: str,
    nome_fn=None,
) -> tuple[list[str], np.ndarray]:
    imagens = sorted(
        p for p in catalog_dir.iterdir()
        if p.suffix.lower() in EXTENSOES_SUPORTADAS
    )
    nomes, embs = [], []
    for img_path in imagens:
        pil    = Image.open(img_path).convert("RGB")
        tensor = preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        nome = nome_fn(img_path.stem) if nome_fn else img_path.stem
        nomes.append(nome)
        embs.append(emb.cpu().float().numpy()[0])
        print(f"  [catalogo] {nome}")
    return nomes, np.array(embs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Embedding do recorte ao vivo
# ---------------------------------------------------------------------------

def embed_recorte(recorte_bgr: np.ndarray, model, preprocess, device: str) -> np.ndarray:
    pil    = Image.fromarray(cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB))
    tensor = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()[0]


# ---------------------------------------------------------------------------
# Detector 1 — diferenca de fundo
# ---------------------------------------------------------------------------

def capturar_fundo(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, FUNDO_BLUR_KERNEL, 0)


def detectar_produto_fundo(
    frame: np.ndarray,
    fundo: np.ndarray,
    area_min: float,
) -> tuple[int, int, int, int] | None:
    h_frame, w_frame = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, FUNDO_BLUR_KERNEL, 0)
    diff       = cv2.absdiff(fundo, gray)
    _, mascara = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kern    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kern)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kern)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    validos = [c for c in contornos if cv2.contourArea(c) >= area_min]
    if not validos:
        return None
    bx, by, bw, bh = cv2.boundingRect(max(validos, key=cv2.contourArea))
    dx = int(bw * MARGEM_BOX);  dy = int(bh * MARGEM_BOX)
    x1 = max(0, bx - dx);       y1 = max(0, by - dy)
    x2 = min(w_frame, bx + bw + dx);  y2 = min(h_frame, by + bh + dy)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Detector 2 — YOLO-World (vocabulario aberto)
# ---------------------------------------------------------------------------

def carregar_detector_yolo():
    try:
        from ultralytics import YOLO  # lazy import — opcional
    except ImportError:
        print(
            "AVISO: pacote 'ultralytics' nao encontrado.\n"
            "  Para instalar: .venv/bin/pip install ultralytics\n"
            "  Continuando com detector por diferenca de fundo."
        )
        return None
    try:
        modelo = YOLO(YOLO_MODELO)
        modelo.set_classes(YOLO_CLASSES + YOLO_CLASSES_FRUTA)
        return modelo
    except Exception as exc:
        print(
            f"AVISO: nao foi possivel carregar {YOLO_MODELO}: {exc}\n"
            "  Continuando com detector por diferenca de fundo."
        )
        return None


def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _mascara_novidade(frame: np.ndarray, fundo: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, FUNDO_BLUR_KERNEL, 0)
    diff       = cv2.absdiff(fundo, gray)
    _, mascara = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mascara


def detectar_produto_yolo(
    frame: np.ndarray,
    yolo_model,
    fundo_ref,
    diag: dict = None,
    sem_frutas: bool = False,
) -> tuple[int, int, int, int, str, float] | None:
    """
    Roda YOLO-World, filtra pessoas/maos/frutas-invalidas, e escolhe a embalagem
    com maior sobreposicao com a mascara de novidade (se fundo disponivel) ou maior
    confianca.  Retorna (x1, y1, x2, y2, classe, confianca) ou None.
    Se diag for passado, preenche {n_total, n_desc_classe, n_desc_pessoa, n_desc_area}.
    """
    h, w = frame.shape[:2]
    results = yolo_model(frame, verbose=False, conf=YOLO_CONF)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    deteccoes = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
        conf   = float(boxes.conf[i])
        idx_cl = int(boxes.cls[i])
        _all = YOLO_CLASSES + YOLO_CLASSES_FRUTA
        classe = _all[idx_cl] if idx_cl < len(_all) else "unknown"
        deteccoes.append((x1, y1, x2, y2, conf, classe))

    n_total     = len(deteccoes)
    pessoas     = [(x1, y1, x2, y2)
                   for x1, y1, x2, y2, _, cl in deteccoes
                   if cl in YOLO_DESCARTAR]
    n_desc_cls  = len(pessoas)
    produtos    = [(x1, y1, x2, y2, conf, cl)
                   for x1, y1, x2, y2, conf, cl in deteccoes
                   if cl not in YOLO_DESCARTAR]

    n_antes_iou = len(produtos)
    if pessoas:
        produtos = [
            d for d in produtos
            if all(_iou(d[:4], p) <= YOLO_IOU_PESSOA for p in pessoas)
        ]
    n_desc_pessoa = n_antes_iou - len(produtos)

    if not produtos:
        if diag is not None:
            diag.update(n_total=n_total, n_desc_classe=n_desc_cls,
                        n_desc_pessoa=n_desc_pessoa, n_desc_area=0)
        return None

    # Descarta caixas maiores que YOLO_MAX_AREA_FRAC (troncos, pessoas mal filtradas)
    area_max    = YOLO_MAX_AREA_FRAC * h * w
    n_antes_area = len(produtos)
    produtos    = [d for d in produtos if (d[2] - d[0]) * (d[3] - d[1]) <= area_max]
    n_desc_area = n_antes_area - len(produtos)

    if diag is not None:
        diag.update(n_total=n_total, n_desc_classe=n_desc_cls,
                    n_desc_pessoa=n_desc_pessoa, n_desc_area=n_desc_area)

    if not produtos:
        return None

    # Filtros adicionais para frutas/hortifruti
    if sem_frutas:
        # Remove todas as caixas de fruta quando a via esta desativada
        produtos = [d for d in produtos if d[5] not in _YOLO_FRUTA_SET]
    else:
        embalagens = [(d[0], d[1], d[2], d[3])
                      for d in produtos if d[5] not in _YOLO_FRUTA_SET]
        area_max_fruta = YOLO_MAX_AREA_FRUTA * h * w
        def _fruta_valida(det):
            cl   = det[5]
            area = (det[2] - det[0]) * (det[3] - det[1])
            if cl not in _YOLO_FRUTA_SET:
                return True
            if area > area_max_fruta:
                return False
            if any(_iou(det[:4], emb) > YOLO_IOU_FRUTA_EMB for emb in embalagens):
                return False
            return True
        produtos = [d for d in produtos if _fruta_valida(d)]

    if not produtos:
        return None

    if fundo_ref is not None:
        msk = _mascara_novidade(frame, fundo_ref)
        def _nov(det):
            bx1, by1, bx2, by2 = max(0, det[0]), max(0, det[1]), min(w, det[2]), min(h, det[3])
            if bx2 <= bx1 or by2 <= by1:
                return 0.0
            return float(msk[by1:by2, bx1:bx2].mean()) / 255.0
        # novidade DESC, confianca DESC, area ASC (menor embalagem preferida em empate)
        produtos.sort(
            key=lambda d: (_nov(d), d[4], -((d[2] - d[0]) * (d[3] - d[1]))),
            reverse=True,
        )
    else:
        # confianca DESC, area ASC
        produtos.sort(
            key=lambda d: (d[4], -((d[2] - d[0]) * (d[3] - d[1]))),
            reverse=True,
        )

    bx1, by1, bx2, by2, conf, classe = produtos[0]
    dx  = int((bx2 - bx1) * MARGEM_BOX);  dy = int((by2 - by1) * MARGEM_BOX)
    bx1 = max(0, bx1 - dx);  by1 = max(0, by1 - dy)
    bx2 = min(w, bx2 + dx);  by2 = min(h, by2 + dy)
    return bx1, by1, bx2, by2, classe, conf


# ---------------------------------------------------------------------------
# Geometria da caixa
# ---------------------------------------------------------------------------

def bbox_para_clip(
    x1: int, y1: int, x2: int, y2: int,
    h_frame: int, w_frame: int,
) -> tuple[int, int, int, int]:
    dw   = x2 - x1;  dh = y2 - y1
    lado = max(dw, dh, 1)
    lado = min(lado, h_frame, w_frame)
    cx   = (x1 + x2) // 2;  cy = (y1 + y2) // 2
    qx1  = max(0, min(cx - lado // 2, w_frame - lado))
    qy1  = max(0, min(cy - lado // 2, h_frame - lado))
    return qx1, qy1, qx1 + lado, qy1 + lado


def suavizar_caixa(
    atual: tuple | None,
    nova:  tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    if atual is None:
        return tuple(float(v) for v in nova)
    a = SUAVIZACAO_ALPHA
    return tuple(a * n + (1 - a) * c for c, n in zip(atual, nova))


def caixa_central(h: int, w: int) -> tuple[int, int, int, int]:
    lado = min(h, w) // 2
    cx, cy = w // 2, h // 2
    x1 = cx - lado // 2;  y1 = cy - lado // 2
    return x1, y1, x1 + lado, y1 + lado


def mover_caixa(caixa, dx, dy, h, w):
    x1, y1, x2, y2 = caixa
    lw = x2 - x1;  lh = y2 - y1
    x1 = max(0, min(x1 + dx, w - lw))
    y1 = max(0, min(y1 + dy, h - lh))
    return x1, y1, x1 + lw, y1 + lh


def redimensionar_caixa(caixa, delta, h, w):
    x1, y1, x2, y2 = caixa
    cx   = (x1 + x2) // 2;  cy = (y1 + y2) // 2
    lado = max(20, min(max(x2 - x1, y2 - y1) + delta, min(h, w)))
    x1   = max(0, min(cx - lado // 2, w - lado))
    y1   = max(0, min(cy - lado // 2, h - lado))
    return x1, y1, x1 + lado, y1 + lado


# ---------------------------------------------------------------------------
# Trilha de objeto unico (usada pelos modos area e linha)
# ---------------------------------------------------------------------------

def _trilha_nova():
    return [], [], "", 0.0, 0, False
    # (centroides, areas, melhor_nome, melhor_sim, frames_sem_det, cruzou)


def _decidir_direcao_area(areas):
    if len(areas) < TRILHA_MIN_FRAMES:
        return None
    arr   = np.array(areas, dtype=float)
    suave = np.array([
        float(np.mean(arr[max(0, i - AREA_JANELA_SUAVE + 1): i + 1]))
        for i in range(len(arr))
    ])
    n3         = max(1, len(suave) // 3)
    med_inicio = float(np.median(suave[:n3]))
    med_fim    = float(np.median(suave[-n3:]))
    if med_inicio <= 0:
        return None
    razao = med_fim / med_inicio
    if razao >= AREA_FATOR_ENTRADA:
        return "entrada" if APROXIMANDO_EH_ENTRADA else "saida"
    if razao <= AREA_FATOR_SAIDA:
        return "saida"   if APROXIMANDO_EH_ENTRADA else "entrada"
    return None


# ---------------------------------------------------------------------------
# Eventos de inventario
# ---------------------------------------------------------------------------

def _processar_evento(
    direcao: str,
    produto: str,
    similaridade: float,
    inventario: dict,
    eventos: list,
    modo_evento: str = "manual",
    origem: str = "clip",
):
    """
    Registra evento, atualiza inventario, retorna (feedback_texto, feedback_cor).
    modo_evento: "reconhecimento" | "area" | "linha" | "manual"
    origem: "clip" | "detector" | "manual"
    """
    ts          = datetime.now().isoformat(timespec="seconds")
    produto_log = produto or "nao_reconhecido"
    eventos.append({
        "timestamp":    ts,
        "produto":      produto_log,
        "direcao":      direcao,
        "similaridade": round(similaridade, 4),
        "modo":         modo_evento,
        "origem":       origem,
    })

    if not produto:
        txt = ("ENTRADA  (nao reconhecido)"
               if direcao == "entrada" else
               "SAIDA  (nao reconhecido)")
        return txt, COR_CINZA

    if direcao == "entrada":
        inventario[produto] = inventario.get(produto, 0) + 1
        return f"+1  {produto}", COR_VERDE

    atual = inventario.get(produto, 0)
    if atual <= 0:
        print(f"[AVISO] Saida de '{produto}' com qtd={atual} — inconsistencia registrada")
        return f"? {produto}  (inconsist.)", COR_AMARELO
    inventario[produto] = atual - 1
    if inventario[produto] == 0:
        del inventario[produto]
    return f"-1  {produto}", COR_VERMELHO


# ---------------------------------------------------------------------------
# Persistencia do inventario
# ---------------------------------------------------------------------------

def salvar_inventario(inventario, eventos, pasta):
    pasta.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = pasta / f"inventario_{ts}.json"
    json_path.write_text(
        json.dumps(
            {"timestamp": ts, "inventario": inventario, "eventos": eventos},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = pasta / f"eventos_{ts}.csv"
    _campos = ["timestamp", "produto", "direcao", "similaridade", "modo", "origem"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(eventos)

    return json_path, csv_path


def imprimir_resumo(inventario, eventos):
    print("\n=== INVENTARIO FINAL ===")
    if not inventario:
        print("  (vazio)")
    else:
        for nome, qtd in sorted(inventario.items()):
            print(f"  {qtd:3d}x  {nome}")
    print(f"\n=== EVENTOS ({len(eventos)} registrado(s)) ===")
    for ev in eventos:
        sinal = "+" if ev["direcao"] == "entrada" else "-"
        modo  = ev.get("modo", "?")
        print(f"  {ev['timestamp']}  {sinal}  {ev['produto']}"
              f"  (sim={ev['similaridade']:.3f}  modo={modo})")


# ---------------------------------------------------------------------------
# Primitivos de desenho
# ---------------------------------------------------------------------------

def _fundo_semitransparente(frame, x1, y1, x2, y2, alpha):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COR_PRETO, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def texto_com_fundo(
    frame: np.ndarray,
    texto: str,
    x: int, y: int,
    cor: tuple,
    escala: float = 0.7,
    espessura: int = 2,
):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(texto, fonte, escala, espessura)
    pad = 5
    _fundo_semitransparente(
        frame,
        x - pad, y - th - pad,
        x + tw + pad, y + base + pad,
        ALPHA_FUNDO,
    )
    cv2.putText(frame, texto, (x, y), fonte, escala, cor, espessura, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Elementos de overlay compostos
# ---------------------------------------------------------------------------

def desenhar_linha_cruzamento(frame, linha_x):
    h = frame.shape[0]
    cv2.line(frame, (linha_x, 0), (linha_x, h), COR_AZUL, 2)
    texto_com_fundo(frame, "L: linha", linha_x + 6, 52, COR_AZUL, escala=0.48, espessura=1)
    mid = h // 2
    cv2.arrowedLine(frame, (linha_x - 35, mid - 18), (linha_x - 6, mid - 18),
                    COR_VERDE,    1, tipLength=0.4)
    cv2.arrowedLine(frame, (linha_x + 6, mid + 18),  (linha_x + 35, mid + 18),
                    COR_VERMELHO, 1, tipLength=0.4)


def desenhar_indicador_area(frame, area_atual, razao_atual):
    x      = 10
    y_base = 80
    if area_atual is not None:
        texto_com_fundo(frame, f"area: {area_atual:.4f}",
                        x, y_base, COR_BRANCO, escala=0.50, espessura=1)
    else:
        texto_com_fundo(frame, "area: --",
                        x, y_base, COR_CINZA,  escala=0.50, espessura=1)
    if razao_atual is not None:
        if razao_atual >= AREA_FATOR_ENTRADA:
            cor_r = COR_VERDE;    tag = "APROX"
        elif razao_atual <= AREA_FATOR_SAIDA:
            cor_r = COR_VERMELHO; tag = "AFAS"
        else:
            cor_r = COR_BRANCO;   tag = "neutro"
        texto_com_fundo(frame, f"razao: {razao_atual:.2f}x  {tag}",
                        x, y_base + 22, cor_r, escala=0.50, espessura=1)
    else:
        texto_com_fundo(frame, "razao: --",
                        x, y_base + 22, COR_CINZA, escala=0.50, espessura=1)


def desenhar_trilha(frame, centroides):
    if not centroides:
        return
    recentes = centroides[-30:]
    for i, pt in enumerate(recentes):
        raio = max(2, 2 + i * 2 // max(len(recentes), 1))
        cv2.circle(frame, pt, raio, COR_AMARELO, -1)
    for i in range(1, len(recentes)):
        cv2.line(frame, recentes[i - 1], recentes[i], COR_AMARELO, 1)


def desenhar_inventario(frame, inventario):
    h, w = frame.shape[:2]
    if w < 300:
        return
    painel_x    = w - 215
    n           = max(1, len(inventario))
    painel_base = min(h - 70, 45 + 26 * n + 10)
    _fundo_semitransparente(frame, painel_x - 8, 8, w - 8, painel_base, 0.65)
    texto_com_fundo(frame, "INVENTARIO", painel_x, 32,
                    COR_BRANCO, escala=0.56, espessura=2)
    if not inventario:
        texto_com_fundo(frame, "(vazio)", painel_x, 58,
                        COR_CINZA, escala=0.50, espessura=1)
    else:
        for i, (nome, qtd) in enumerate(sorted(inventario.items())):
            cor_item = COR_VERDE if qtd > 0 else COR_CINZA
            texto_com_fundo(
                frame, f"{qtd}x  {nome[:17]}",
                painel_x, 58 + i * 26,
                cor_item, escala=0.50, espessura=1,
            )


def desenhar_feedback(frame, texto, cor, ate):
    if not texto or time.time() > ate:
        return
    h, w = frame.shape[:2]
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala, esp = 1.0, 3
    (tw, th), base = cv2.getTextSize(texto, fonte, escala, esp)
    x = max(10, (w - tw) // 2)
    y = h * 2 // 3
    _fundo_semitransparente(frame, x - 12, y - th - 12, x + tw + 12, y + base + 12, 0.78)
    cv2.putText(frame, texto, (x, y), fonte, escala, cor, esp, cv2.LINE_AA)


def desenhar_frame(
    frame: np.ndarray,
    guia: tuple[int, int, int, int],
    nome: str,
    similaridade: float,
    melhor_palpite: str,
    limiar: float,
    n_produtos: int,
    modo: str,             # "auto" | "sem_fundo" | "fallback" | "sem_objeto" | "manual"
    com_enriquecido: bool,
    direcao_flag: str,     # "nenhum" | "area" | "linha"
    modo_direcao: str,     # "area" | "linha" — apenas relevante quando direcao_flag != "nenhum"
    detector: str,         # "fundo" | "yolo"
    det_info: str = "",       # "classe  0.XX" quando YOLO
    reconhec_info: str = "",  # "voto X N/K" ou "bloq X N/M" no modo reconhecimento
    via_ident: str = "",      # "detector" | "catalogo" — via da ultima inferencia
    decisao_atual: str = "",  # "aceito" | "ambiguo" | "rejeitado_limiar" | ""
    margem_minima: float = MARGEM_MINIMA,
):
    h = frame.shape[0]
    x1, y1, x2, y2 = guia

    # Cor da borda — reflecte a decisao
    if modo == "sem_objeto":
        cor_guia = COR_CINZA
    elif modo in ("sem_fundo", "fallback"):
        cor_guia = COR_AMARELO
    elif modo == "manual" or not nome:
        cor_guia = COR_CINZA
    elif via_ident == "detector":
        cor_guia = COR_VERDE
    elif decisao_atual == "aceito":
        cor_guia = COR_VERDE
    elif decisao_atual == "ambiguo":
        cor_guia = COR_AMARELO
    else:
        cor_guia = COR_VERMELHO

    cv2.rectangle(frame, (x1, y1), (x2, y2), cor_guia, 2)

    # Resultado CLIP abaixo da caixa
    if modo == "sem_objeto":
        texto_com_fundo(frame, "sem objeto", x1, y2 + 28, COR_CINZA, escala=0.65)
    elif not nome:
        if modo == "sem_fundo":
            texto_com_fundo(frame, "Pressione [B] com a cena vazia",
                            x1, y2 + 28, COR_AMARELO, escala=0.65)
        else:
            texto_com_fundo(frame, "aguardando...", x1, y2 + 28, COR_CINZA, escala=0.65)
    else:
        if via_ident == "detector":
            linha1 = f"{nome} (detector)"
            linha2 = f"conf: {similaridade:.2f}"
            cor1   = COR_VERDE
        elif decisao_atual == "aceito":
            linha1 = f"{nome} (catalogo {similaridade:.2f})"
            linha2 = ""
            cor1   = COR_VERDE
        elif decisao_atual == "ambiguo":
            linha1 = f"ambiguo: {melhor_palpite}"
            linha2 = f"margem insuf.  ({similaridade:.3f})"
            cor1   = COR_AMARELO
        else:
            linha1 = "NAO RECONHECIDO"
            linha2 = f"melhor: {melhor_palpite}  ({similaridade:.3f})"
            cor1   = COR_VERMELHO
        texto_com_fundo(frame, linha1, x1, y2 + 30, cor1,      escala=0.85, espessura=2)
        if linha2:
            texto_com_fundo(frame, linha2, x1, y2 + 58, COR_CINZA, escala=0.60, espessura=1)

    # Indicador de modo de deteccao (canto superior esquerdo)
    if   modo == "auto":       modo_txt, modo_cor = "AUTO",      COR_VERDE
    elif modo == "sem_fundo":  modo_txt, modo_cor = "SEM FUNDO", COR_AMARELO
    elif modo == "fallback":   modo_txt, modo_cor = "FALLBACK",  COR_AMARELO
    elif modo == "sem_objeto": modo_txt, modo_cor = "SEM OBJ",   COR_CINZA
    else:                      modo_txt, modo_cor = "MANUAL",    COR_CINZA
    texto_com_fundo(frame, f"D: {modo_txt}", 10, 22, modo_cor, escala=0.55, espessura=1)

    # YOLO: classe e confianca
    if det_info:
        texto_com_fundo(frame, det_info, 10, 50, COR_AZUL, escala=0.50, espessura=1)

    # Modo reconhecimento: contador de estabilidade (y=78, nao colide com area/razao)
    if reconhec_info:
        cor_ri = COR_AMARELO if "bloq" not in reconhec_info else COR_CINZA
        texto_com_fundo(frame, reconhec_info, 10, 78, cor_ri, escala=0.52, espessura=1)

    # Rodape (tres linhas)
    cat_txt = "+enriq" if com_enriquecido else "normal"
    det_txt = detector
    if direcao_flag == "nenhum":
        modo_inv_txt = "reconhecimento"
    else:
        modo_inv_txt = f"direcao:{modo_direcao}"
    s1 = (f"Limiar: {limiar:.2f}  marg: {margem_minima:.3f}  |  {n_produtos} prod  |  "
          f"cat: {cat_txt}  |  det: {det_txt}  |  {modo_inv_txt}")
    s2 = "[+/-] limiar  [</>] margem  [B] fundo  [D] auto  [L] area/linha  [R] reset"
    s3 = "[A]+prod  [S]-prod  [C] enrich  [SPC] teste  [Q] sair"
    texto_com_fundo(frame, s1, 10, h - 55, COR_BRANCO, escala=0.46, espessura=1)
    texto_com_fundo(frame, s2, 10, h - 32, COR_CINZA,  escala=0.42, espessura=1)
    texto_com_fundo(frame, s3, 10, h - 10, COR_CINZA,  escala=0.42, espessura=1)


# ---------------------------------------------------------------------------
# Salvar recortes
# ---------------------------------------------------------------------------

def _proximo_indice(pasta, rotulo, sufixo):
    n = 1
    while (pasta / f"{rotulo}{sufixo}{n}.jpg").exists():
        n += 1
    return n


def salvar_recorte(recorte_bgr, testes_dir):
    print("\n[ESPACO] Rotulo do produto (sem espacos nem acentos):")
    rotulo = input("  > ").strip()
    if not rotulo:
        print("  Rotulo vazio — recorte descartado.")
        return
    n       = _proximo_indice(testes_dir, rotulo, "_")
    destino = testes_dir / f"{rotulo}_{n}.jpg"
    cv2.imwrite(str(destino), recorte_bgr)
    print(f"  Salvo: {destino}")


def salvar_enriquecido(recorte_bgr, enriquecido_dir, sugestao):
    prompt = (f"\n[C] Rotulo do produto [{sugestao}]:" if sugestao
              else "\n[C] Rotulo do produto:")
    print(prompt)
    entrada = input("  > ").strip()
    rotulo  = entrada or sugestao
    if not rotulo:
        print("  Rotulo vazio — recorte descartado.")
        return
    n       = _proximo_indice(enriquecido_dir, rotulo, "_enr_")
    destino = enriquecido_dir / f"{rotulo}_enr_{n}.jpg"
    cv2.imwrite(str(destino), recorte_bgr)
    print(f"  Salvo: {destino}")


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def abrir_camera(indice):
    cap = cv2.VideoCapture(indice, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(indice)


def aquecer_camera(cap, n=AQUECIMENTO_FRAMES):
    for _ in range(n):
        cap.read()


def checar_imagem_escura(cap, indice):
    ret, frame = cap.read()
    if not ret:
        return
    media = float(frame.mean())
    if media < LIMIAR_ESCURIDAO:
        print(
            f"\nAVISO: camera {indice} parece nao estar entregando imagem "
            f"(media de pixels = {media:.1f}).\n"
            f"Tente outro indice com --camera N  (ex.: --camera 1).\n"
        )


def _cameras_av() -> list[tuple[int, str]]:
    """
    Lista dispositivos de video via AVFoundation (pyobjc).
    A ordem coincide com os indices do backend CAP_AVFOUNDATION do OpenCV.
    Levanta ImportError se pyobjc-framework-AVFoundation nao estiver instalado.
    """
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo  # type: ignore
    devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
    return [(i, str(d.localizedName())) for i, d in enumerate(devices)]


def listar_cameras():
    print("Varrendo dispositivos de camera (0 a 3)...\n")
    try:
        av_nomes = {i: n for i, n in _cameras_av()}
        print("  Nomes AVFoundation:")
        for i, n in sorted(av_nomes.items()):
            print(f"    [{i}] {n}")
        print()
    except ImportError:
        av_nomes = {}

    for i in range(4):
        cap      = abrir_camera(i)
        nome_str = f"  ({av_nomes[i]})" if i in av_nomes else ""
        if not cap.isOpened():
            print(f"  [{i}]{nome_str}  nao disponivel")
            cap.release()
            continue
        aquecer_camera(cap, AQUECIMENTO_FRAMES)
        ret, frame = cap.read()
        if not ret:
            print(f"  [{i}]{nome_str}  abriu mas nao entregou frame apos aquecimento")
            cap.release()
            continue
        h, w  = frame.shape[:2]
        media = float(frame.mean())
        aviso = "  <- parece preta" if media < LIMIAR_ESCURIDAO else ""
        print(f"  [{i}]{nome_str}  {w}x{h}  media pixels: {media:5.1f}{aviso}")
        cap.release()
    print()


# ---------------------------------------------------------------------------
# VLM local (Ollama) — usado com --vlm
# ---------------------------------------------------------------------------

def _chamar_ollama_vlm(recorte_bgr: np.ndarray, nomes: list) -> str:
    """Envia recorte BGR ao Ollama e devolve nome exato da lista ou 'nenhum'."""
    _, buf   = cv2.imencode(".jpg", recorte_bgr)
    img_b64  = base64.b64encode(buf.tobytes()).decode()
    lista    = "\n".join(f"- {n}" for n in nomes)
    prompt   = (
        "You are analyzing a food product image from a fridge camera.\n"
        "Choose exactly one name from the catalog list below that matches "
        "the product in the image.\n"
        "If none match, respond: nenhum\n"
        "Respond with ONLY the exact name or 'nenhum'. "
        "No explanation, no punctuation, nothing else.\n\n"
        f"Catalog:\n{lista}"
    )
    payload = json.dumps({
        "model":   OLLAMA_MODELO_VLM,
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
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_VLM) as resp:
        data = json.loads(resp.read())
    resposta  = data.get("response", "").strip().strip("\"'")
    nomes_set = set(nomes)
    if resposta.lower() == "nenhum":
        return "nenhum"
    if resposta in nomes_set:
        return resposta
    resp_lower = resposta.lower()
    for n in nomes:
        if n.lower() == resp_lower:
            return n
    return f"invalida:{resposta[:40]}"


def _thread_vlm(
    recorte_bgr: np.ndarray,
    clip_opiniao: str,
    nomes_list: list,
    vlm_lock: threading.Lock,
    vlm_resultado: dict,
    dados_evento: dict,
) -> None:
    try:
        vlm = _chamar_ollama_vlm(recorte_bgr, nomes_list)
    except Exception as exc:
        print(f"\n  [vlm] erro: {exc}")
        vlm = ""
    with vlm_lock:
        vlm_resultado["pronto"]       = True
        vlm_resultado["vlm"]          = vlm
        vlm_resultado["clip"]         = clip_opiniao
        vlm_resultado["dados_evento"] = dados_evento


# ---------------------------------------------------------------------------
# Diagnostico de sessao
# ---------------------------------------------------------------------------

def _salvar_crop_log(
    recorte: np.ndarray,
    crops_dir: Path,
    prefixo: str,
    n: int,
    produto: str,
) -> None:
    if recorte is None or recorte.size == 0:
        return
    nome_safe = re.sub(r"[^\w\-]", "_", produto)[:40]
    destino   = crops_dir / f"{prefixo}_{n:04d}_{nome_safe}.jpg"
    cv2.imwrite(str(destino), recorte)


def _log_linha(
    log_f,
    contador_frames: int,
    modo: str,
    yolo_classe: str,
    yolo_conf: float,
    x1: int, y1: int, x2: int, y2: int,
    h_frame: int, w_frame: int,
    diag_yolo: dict,
    via_ident: str,
    top3: list,
    margem: float,
    decisao: str,
    passagem_ativa: bool,
    passagem_acertos: list,
    reconhec_bloqueado: dict,
) -> None:
    ts        = datetime.now().strftime("%H:%M:%S.%f")[:12]
    bw        = x2 - x1;  bh = y2 - y1
    area_frac = bw * bh / max(1, h_frame * w_frame)
    desc_p    = diag_yolo.get("n_desc_pessoa", 0)
    desc_a    = diag_yolo.get("n_desc_area",   0)
    det_str   = f"{yolo_classe}/{yolo_conf:.2f}" if yolo_classe else "-/-"

    if via_ident == "detector":
        cat_str = f"via=detector {top3[0][0]}/{top3[0][1]:.2f}" if top3 else "via=detector"
    else:
        cat_str = " | ".join(f"{nm}/{s:.3f}" for nm, s in top3) if top3 else "-"

    if passagem_ativa:
        prod_cnts: dict = {}
        for p, _, _ in passagem_acertos:
            prod_cnts[p] = prod_cnts.get(p, 0) + 1
        pass_str = " ".join(f"{p}:{c}" for p, c in sorted(prod_cnts.items()))
        pass_info = pass_str if pass_str else "aguard"
    else:
        pass_info = "-"
    cool_str = (
        " ".join(f"{p}:{c}" for p, c in sorted(reconhec_bloqueado.items()))
        if reconhec_bloqueado else "-"
    )
    marg_str = f"{margem:.3f}" if via_ident != "detector" else "-"

    log_f.write(
        f"{ts}  fr={contador_frames:5d}  {modo:<10}"
        f"  det={det_str:<22}  box={x1},{y1},{bw},{bh}"
        f"  area={area_frac:.3f}  desc={desc_p}p+{desc_a}a"
        f"  |  {cat_str}"
        f"  marg={marg_str}  dec={decisao}"
        f"  |  pass=[{pass_info}]  cool=[{cool_str}]\n"
    )


def _log_resumo(
    log_f,
    n_total: int,
    n_aceitas: int,
    n_rej_limiar: int,
    n_ambiguas: int,
    ambiguas_pares: dict,
) -> None:
    log_f.write("\n" + "=" * 70 + "\n")
    log_f.write("RESUMO DA SESSAO\n")
    log_f.write(f"  Total de inferencias : {n_total}\n")
    log_f.write(f"  Aceitas              : {n_aceitas}\n")
    log_f.write(f"  Rejeitadas (limiar)  : {n_rej_limiar}\n")
    log_f.write(f"  Ambiguas (margem)    : {n_ambiguas}\n")
    if ambiguas_pares:
        log_f.write("  Pares ambiguos mais frequentes:\n")
        for (p1, p2), cnt in sorted(
                ambiguas_pares.items(), key=lambda kv: kv[1], reverse=True):
            log_f.write(f"    {cnt:4d}x  {p1}  vs  {p2}\n")
    log_f.write("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogo", default=None)
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--camera",   type=int, default=None,
                        help="Indice da camera (padrao: escolha automatica pelo primeiro indice "
                             "com media de pixels > 20)")
    parser.add_argument("--camera-nome", default=None, metavar="TEXTO",
                        help="Seleciona a primeira camera cujo nome AVFoundation contenha TEXTO "
                             "(sem diferenciar maiusculas). Requer pyobjc-framework-AVFoundation.")
    parser.add_argument("--com-enriquecido", action="store_true")
    parser.add_argument("--listar-cameras",  action="store_true")
    parser.add_argument("--sem-frutas", action="store_true",
                        help="Desativa a via de frutas/hortifruti (so usa catalogo CLIP)")
    parser.add_argument("--vlm", action="store_true",
                        help="Verificar cada passagem com VLM local "
                             f"(Ollama {OLLAMA_MODELO_VLM} em {OLLAMA_URL})")
    parser.add_argument(
        "--detector",
        choices=["fundo", "yolo"],
        default="fundo",
        help="'fundo': diferenca de fundo (padrao)  |  'yolo': YOLO-World vocabulario aberto",
    )
    parser.add_argument(
        "--direcao",
        choices=["nenhum", "area", "linha"],
        default="nenhum",
        help=(
            "'nenhum': adicao por estabilidade de reconhecimento (padrao)  |  "
            "'area': trilha por tamanho aparente  |  "
            "'linha': cruzamento de linha vertical"
        ),
    )
    args = parser.parse_args()

    if args.listar_cameras:
        listar_cameras()
        return

    config          = load_config(args.config)
    catalog_dir     = Path(args.catalogo or config["download"]["catalog_dir"])
    testes_dir      = Path("testes")
    enriquecido_dir = Path("catalogo_enriquecido")
    inventario_dir  = Path("inventario")
    testes_dir.mkdir(exist_ok=True)
    enriquecido_dir.mkdir(exist_ok=True)

    if not catalog_dir.is_dir():
        sys.exit(f"Pasta de catalogo nao encontrada: {catalog_dir}")

    imagens = [p for p in catalog_dir.iterdir()
               if p.suffix.lower() in EXTENSOES_SUPORTADAS]
    if not imagens:
        sys.exit(
            f"Catalogo vazio em '{catalog_dir}'.\n"
            "Execute primeiro:  python baixar_catalogo.py"
        )

    device = get_device()
    print(f"Dispositivo : {device}")
    print(f"Modelo CLIP : {MODELO_CLIP} / {PRETRAINED}")
    print("Carregando modelo CLIP...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        MODELO_CLIP, pretrained=PRETRAINED, device=device
    )
    clip_model.eval()

    print(f"Indexando {len(imagens)} imagem(ns) do catalogo...")
    nomes, embeddings = carregar_catalogo(catalog_dir, clip_model, preprocess, device)

    com_enriquecido = args.com_enriquecido
    if com_enriquecido:
        imgs_enr = ([p for p in enriquecido_dir.iterdir()
                     if p.suffix.lower() in EXTENSOES_SUPORTADAS]
                    if enriquecido_dir.is_dir() else [])
        if imgs_enr:
            print(f"Indexando {len(imgs_enr)} referencia(s) de enriquecimento...")
            nomes_enr, embs_enr = carregar_catalogo(
                enriquecido_dir, clip_model, preprocess, device,
                nome_fn=lambda s: re.sub(r"_enr_\d+$", "", s),
            )
            nomes      = nomes + nomes_enr
            embeddings = np.vstack([embeddings, embs_enr])
        else:
            print("  catalogo_enriquecido/ vazio ou inexistente, ignorado.")

    print(f"Pronto — {len(nomes)} produto(s) carregado(s).\n")

    # ---- Detector de localizacao ----
    detector   = args.detector
    yolo_model = None
    if detector == "yolo":
        print(f"Carregando YOLO-World ({YOLO_MODELO})...")
        yolo_model = carregar_detector_yolo()
        if yolo_model is None:
            detector = "fundo"
        else:
            print(f"  Classes: {YOLO_CLASSES}\n")

    sem_frutas = args.sem_frutas
    print(f"Detector ativo   : {detector.upper()}"
          + ("  (frutas desativadas)" if sem_frutas else ""))

    # ---- Modo de inventario ----
    direcao_flag  = args.direcao                   # "nenhum" | "area" | "linha"
    direcao_ativa = (direcao_flag != "nenhum")
    modo_direcao  = direcao_flag if direcao_ativa else "area"  # estado interno do [l]

    print(f"Modo inventario  : {direcao_flag}")

    # ---- VLM local (--vlm) ----
    usar_vlm = args.vlm
    if usar_vlm:
        print(f"VLM              : {OLLAMA_MODELO_VLM} em {OLLAMA_URL}")
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{OLLAMA_URL}/api/tags",
                    headers={"User-Agent": "demo_ao_vivo/1.0"},
                ),
                timeout=2,
            )
            print("  Ollama disponivel. Aquecendo modelo...")
            _dummy = np.zeros((8, 8, 3), dtype=np.uint8)
            _chamar_ollama_vlm(_dummy, nomes[:3])
            print("  VLM pronto.")
        except Exception as exc:
            print(f"  AVISO: Ollama nao encontrado ({exc}) — --vlm desativado.")
            usar_vlm = False
    print()

    if args.camera is not None:
        indice_camera = args.camera
        print(f"Abrindo camera {indice_camera} (forcado via --camera)...")
        cap = abrir_camera(indice_camera)
        if not cap.isOpened():
            sys.exit(
                f"Nao foi possivel abrir a camera {indice_camera}.\n"
                "Use --listar-cameras para ver os dispositivos disponiveis."
            )
        print(f"  Aquecendo ({AQUECIMENTO_FRAMES} frames)...")
        aquecer_camera(cap, AQUECIMENTO_FRAMES)
        checar_imagem_escura(cap, indice_camera)
    elif args.camera_nome is not None:
        try:
            av_devices = _cameras_av()
        except ImportError:
            sys.exit(
                "pyobjc-framework-AVFoundation nao instalado — impossivel buscar por nome.\n"
                "Instale com:  .venv/bin/pip install pyobjc-framework-AVFoundation\n"
                "Ou use --camera N diretamente."
            )
        texto = args.camera_nome.lower()
        match = next(((i, n) for i, n in av_devices if texto in n.lower()), None)
        if match is None:
            print(f"Nenhuma camera com nome contendo '{args.camera_nome}'.")
            print("Cameras disponiveis:")
            for i, n in av_devices:
                print(f"  [{i}] {n}")
            sys.exit(1)
        indice_camera, nome_camera = match
        print(f"Camera selecionada por nome: [{indice_camera}] {nome_camera}")
        cap = abrir_camera(indice_camera)
        if not cap.isOpened():
            sys.exit(f"Nao foi possivel abrir a camera {indice_camera} ({nome_camera}).")
        print(f"  Aquecendo ({AQUECIMENTO_FRAMES} frames)...")
        aquecer_camera(cap, AQUECIMENTO_FRAMES)
        checar_imagem_escura(cap, indice_camera)
    else:
        print("Buscando camera automaticamente (indices 0-3)...")
        cap            = None
        indice_camera  = -1
        for i in range(4):
            _cap = abrir_camera(i)
            if not _cap.isOpened():
                print(f"  [{i}] nao disponivel")
                _cap.release()
                continue
            aquecer_camera(_cap, AQUECIMENTO_FRAMES)
            ret, _frame = _cap.read()
            if not ret:
                print(f"  [{i}] abriu mas nao entregou frame — ignorado")
                _cap.release()
                continue
            media = float(_frame.mean())
            if media > LIMIAR_ESCURIDAO:
                print(f"  [{i}] media={media:.1f}  <- escolhida")
                cap           = _cap
                indice_camera = i
                break
            print(f"  [{i}] media={media:.1f}  abaixo do limiar ({LIMIAR_ESCURIDAO}) — ignorada")
            _cap.release()
        if cap is None:
            sys.exit(
                "Nenhuma camera util encontrada (indices 0-3 pretos ou ausentes).\n"
                "Verifique permissoes de camera ou use --camera N explicitamente."
            )

    # ---- estado de inferencia ----
    limiar             = LIMIAR_INICIAL
    margem_minima      = MARGEM_MINIMA
    contador_frames    = 0
    nome_atual         = ""
    similaridade_atual = 0.0
    melhor_palpite     = ""
    via_ident          = ""   # "detector" | "catalogo" — via usada na ultima inferencia
    produto_via:  dict = {}   # {produto: via} — via da ultima vez que o produto foi reconhecido
    # campos computados por inferencia (resetados a cada frame de inferencia)
    decisao_atual:  str  = ""
    top3_atual:     list = []
    margem_atual:   float = 0.0
    diag_yolo:      dict  = {}

    # ---- diagnostico / contadores de sessao ----
    n_log_evento:    int  = 0
    n_inf_total:     int  = 0
    n_inf_aceitas:   int  = 0
    n_inf_rej_limiar: int = 0
    n_inf_ambiguas:  int  = 0
    ambiguas_pares:  dict = {}

    # ---- estado de deteccao ----
    fundo_ref         = None
    deteccao_ativa    = True
    caixa_suavizada   = None
    caixa_manual      = None
    yolo_classe_atual = ""
    yolo_conf_atual   = 0.0

    # ---- rastreio de trilha (area / linha) ----
    (trilha_centroides, trilha_areas,
     trilha_melhor_nome, trilha_melhor_sim,
     trilha_frames_sem_det, trilha_cruzou) = _trilha_nova()
    trilha_melhor_via: str = ""

    # ---- modo reconhecimento (--direcao nenhum): por passagem ----
    passagem_ativa:          bool  = False
    passagem_frames_sem_det: int   = 0
    passagem_frames_total:   int   = 0
    passagem_inf_total:      int   = 0
    passagem_acertos:        list  = []  # (produto, sim, margem) por inferencia aceita
    reconhec_bloqueado:      dict  = {}
    passagem_num:            int   = 0
    passagem_melhor_crop             = None   # np.ndarray | None
    passagem_melhor_sharp:   float  = 0.0

    # ---- estado do VLM em thread ----
    vlm_em_andamento = False
    _vlm_lock        = threading.Lock()
    _vlm_resultado:  dict = {}

    # ---- inventario e feedback ----
    inventario:   dict = {}
    eventos:      list = []
    feedback_texto     = ""
    feedback_cor       = COR_BRANCO
    feedback_ate       = 0.0

    print("Janela aberta.")
    print("  [B] fundo  [D] auto  [L] area/linha  [A/S] manual  [R] reset  [Q] sair")

    # ---- log de sessao ----
    logs_demo_dir    = Path("logs_demo")
    logs_demo_dir.mkdir(exist_ok=True)
    sessao_ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    sessao_crops_dir = logs_demo_dir / sessao_ts
    sessao_crops_dir.mkdir(exist_ok=True)
    capturas_dir     = Path("capturas") / sessao_ts
    capturas_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(
        logs_demo_dir / f"sessao_{sessao_ts}.txt",
        "w", encoding="utf-8", buffering=1,
    )
    log_f.write(f"Sessao: {datetime.now().isoformat(timespec='seconds')}\n")
    log_f.write(
        f"detector={detector}  direcao={direcao_flag}  "
        f"camera={indice_camera}  limiar={LIMIAR_INICIAL}  "
        f"margem={MARGEM_MINIMA}  timeout={PASSAGEM_TIMEOUT}fr  min_acertos={MIN_ACERTOS_PASSAGEM}\n"
    )
    log_f.write("-" * 70 + "\n")
    print(f"Log de sessao: {logs_demo_dir}/sessao_{sessao_ts}.txt")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera encerrou o stream.")
            break

        h, w = frame.shape[:2]
        if caixa_manual is None:
            caixa_manual = caixa_central(h, w)

        # ---- Determinar caixa e modo ----
        diag_yolo = {}
        if deteccao_ativa:
            if detector == "yolo":
                resultado = detectar_produto_yolo(frame, yolo_model, fundo_ref,
                                                  diag=diag_yolo, sem_frutas=sem_frutas)
                if resultado is not None:
                    bx1, by1, bx2, by2, yolo_classe_atual, yolo_conf_atual = resultado
                    caixa_suavizada = suavizar_caixa(
                        caixa_suavizada, (bx1, by1, bx2, by2)
                    )
                    x1, y1, x2, y2 = (int(v) for v in caixa_suavizada)
                    caixa_manual    = (x1, y1, x2, y2)
                    modo            = "auto"
                else:
                    x1, y1, x2, y2   = caixa_central(h, w)
                    yolo_classe_atual = ""
                    yolo_conf_atual   = 0.0
                    modo              = "sem_objeto"  # YOLO sem deteccao: nao roda CLIP
            else:  # "fundo"
                if fundo_ref is None:
                    x1, y1, x2, y2 = caixa_central(h, w)
                    modo = "sem_fundo"
                else:
                    area_min  = h * w * AREA_MINIMA_FRAC
                    detectado = detectar_produto_fundo(frame, fundo_ref, area_min)
                    if detectado is not None:
                        caixa_suavizada = suavizar_caixa(caixa_suavizada, detectado)
                        x1, y1, x2, y2 = (int(v) for v in caixa_suavizada)
                        caixa_manual    = (x1, y1, x2, y2)
                        modo            = "auto"
                    else:
                        x1, y1, x2, y2 = caixa_central(h, w)
                        modo            = "fallback"
        else:
            x1, y1, x2, y2   = caixa_manual
            yolo_classe_atual = ""
            yolo_conf_atual   = 0.0
            modo              = "manual"

        # ---- Recorte quadrado para CLIP ----
        qx1, qy1, qx2, qy2 = bbox_para_clip(x1, y1, x2, y2, h, w)

        # ---- Inferencia: fruta (via detector) ou embalagem (via CLIP) ----
        # Com --detector yolo, modo "sem_objeto" significa nenhuma caixa encontrada:
        # nao rodar o CLIP (evita identificacoes fantasmas no recorte central vazio).
        inferencia_rodou = False
        via_ident        = ""
        decisao_atual    = ""
        top3_atual       = []
        margem_atual     = 0.0
        contador_frames += 1
        _pode_inferir = (modo != "sem_objeto")
        _intervalo_inf = PASSAGEM_INFERENCIA_N if passagem_ativa else INFERENCIA_A_CADA_N_FRAMES
        if _pode_inferir and contador_frames % _intervalo_inf == 0:
            if (not sem_frutas
                    and yolo_classe_atual in _YOLO_FRUTA_SET
                    and yolo_conf_atual >= YOLO_CONF_FRUTA):
                # Fruta: usa o rotulo do YOLO diretamente, sem consultar o catalogo
                melhor_palpite     = YOLO_FRUTA_PT.get(yolo_classe_atual, yolo_classe_atual)
                similaridade_atual = yolo_conf_atual
                nome_atual         = melhor_palpite
                via_ident          = "detector"
                top3_atual         = [(melhor_palpite, yolo_conf_atual)]
                margem_atual       = 1.0   # sem segundo colocado no catalogo
                decisao_atual      = "aceito"
                inferencia_rodou   = True
            else:
                recorte = frame[qy1:qy2, qx1:qx2]
                if recorte.size > 0:
                    query              = embed_recorte(recorte, clip_model, preprocess, device)
                    sims               = embeddings @ query
                    top_idx            = np.argsort(sims)[::-1]
                    top3_atual         = [(nomes[i], float(sims[i])) for i in top_idx[:3]]
                    idx                = int(top_idx[0])
                    melhor_palpite     = nomes[idx]
                    similaridade_atual = float(sims[idx])
                    margem_atual       = (
                        float(sims[top_idx[0]] - sims[top_idx[1]])
                        if len(nomes) >= 2 else 1.0
                    )
                    if similaridade_atual < limiar:
                        decisao_atual = "rejeitado_limiar"
                        nome_atual    = "NAO RECONHECIDO"
                    elif margem_atual < margem_minima:
                        decisao_atual = "ambiguo"
                        nome_atual    = "NAO RECONHECIDO"
                    else:
                        decisao_atual = "aceito"
                        nome_atual    = melhor_palpite
                    via_ident          = "catalogo"
                    inferencia_rodou   = True

        # ---- Contadores de sessao + crops de ambiguo ----
        if inferencia_rodou:
            n_inf_total += 1
            if decisao_atual == "aceito":
                n_inf_aceitas += 1
            elif decisao_atual == "rejeitado_limiar":
                n_inf_rej_limiar += 1
            elif decisao_atual == "ambiguo":
                n_inf_ambiguas += 1
                if len(top3_atual) >= 2:
                    chave = (top3_atual[0][0], top3_atual[1][0])
                    ambiguas_pares[chave] = ambiguas_pares.get(chave, 0) + 1
                _recorte_amb = frame[qy1:qy2, qx1:qx2]
                if _recorte_amb.size > 0:
                    n_log_evento += 1
                    _salvar_crop_log(_recorte_amb, sessao_crops_dir,
                                     "ambiguo", n_log_evento, melhor_palpite)
            _log_linha(
                log_f, contador_frames, modo,
                yolo_classe_atual, yolo_conf_atual,
                x1, y1, x2, y2, h, w,
                diag_yolo, via_ident,
                top3_atual, margem_atual, decisao_atual,
                passagem_ativa, passagem_acertos, reconhec_bloqueado,
            )

        # ---- Modo reconhecimento: decisao por passagem ----
        if not direcao_ativa:
            det_valida = (modo == "auto")

            # Iniciar passagem quando detector encontra produto
            if det_valida and not passagem_ativa:
                passagem_ativa          = True
                passagem_frames_sem_det = 0
                passagem_frames_total   = 0
                passagem_inf_total      = 0
                passagem_acertos        = []
                passagem_melhor_crop    = None
                passagem_melhor_sharp   = 0.0

            if passagem_ativa:
                passagem_frames_total += 1
                if det_valida:
                    passagem_frames_sem_det = 0
                    _rc = frame[qy1:qy2, qx1:qx2]
                    if _rc.size > 0:
                        _gray  = cv2.cvtColor(_rc, cv2.COLOR_BGR2GRAY)
                        _sharp = float(cv2.Laplacian(_gray, cv2.CV_64F).var())
                        if _sharp > passagem_melhor_sharp:
                            passagem_melhor_sharp = _sharp
                            passagem_melhor_crop  = _rc.copy()
                else:
                    passagem_frames_sem_det += 1

                # Acumular inferencias aceitas durante a passagem
                if inferencia_rodou:
                    passagem_inf_total += 1
                    if decisao_atual == "aceito":
                        passagem_acertos.append(
                            (melhor_palpite, similaridade_atual, margem_atual)
                        )
                        produto_via[melhor_palpite] = via_ident

                # Encerrar passagem apos PASSAGEM_TIMEOUT frames sem deteccao
                if passagem_frames_sem_det >= PASSAGEM_TIMEOUT:
                    _clip_opiniao = "nenhum"
                    _decisao_pass = "SEM_REC"
                    _sim_clip     = 0.0

                    if not passagem_acertos:
                        log_f.write(
                            f"  PASSAGEM_SEM_REC  dur={passagem_frames_total}fr"
                            f"  inf={passagem_inf_total}\n"
                        )
                    else:
                        por_prod: dict = {}
                        for prod, sim, marg in passagem_acertos:
                            if prod not in por_prod:
                                por_prod[prod] = {"soma": 0.0, "n": 0, "best": 0.0}
                            por_prod[prod]["soma"] += sim
                            por_prod[prod]["n"]    += 1
                            if sim > por_prod[prod]["best"]:
                                por_prod[prod]["best"] = sim

                        vencedor, dados = max(
                            por_prod.items(), key=lambda kv: kv[1]["soma"]
                        )
                        cnts_str = "  ".join(
                            f"{p}:{d['n']}({d['soma']:.2f})"
                            for p, d in sorted(por_prod.items())
                        )
                        _clip_opiniao = vencedor
                        _sim_clip     = dados["best"]

                        if dados["n"] < MIN_ACERTOS_PASSAGEM:
                            _decisao_pass = "INSUF"
                            log_f.write(
                                f"  PASSAGEM_INSUF  dur={passagem_frames_total}fr"
                                f"  inf={passagem_inf_total}  aceit={len(passagem_acertos)}"
                                f"  [{cnts_str}]  -> {vencedor}\n"
                            )
                        else:
                            _decisao_pass = "OK"
                            if not usar_vlm:
                                feedback_texto, feedback_cor = _processar_evento(
                                    "entrada", vencedor, dados["best"],
                                    inventario, eventos, modo_evento="reconhecimento",
                                    origem=produto_via.get(vencedor, "clip"),
                                )
                                _recorte_ev = frame[qy1:qy2, qx1:qx2]
                                if _recorte_ev.size > 0:
                                    n_log_evento += 1
                                    _salvar_crop_log(_recorte_ev, sessao_crops_dir,
                                                     "evento", n_log_evento, vencedor)
                                feedback_ate = time.time() + FEEDBACK_DURACAO
                            log_f.write(
                                f"  PASSAGEM_OK  dur={passagem_frames_total}fr"
                                f"  inf={passagem_inf_total}  aceit={len(passagem_acertos)}"
                                f"  [{cnts_str}]  -> {vencedor}\n"
                            )

                    # Salvar recorte mais nitido (Change 1)
                    if passagem_frames_total >= PASSAGEM_MIN_FRAMES and passagem_melhor_crop is not None:
                        passagem_num += 1
                        _crop_path = capturas_dir / f"passagem_{passagem_num:03d}.jpg"
                        cv2.imwrite(str(_crop_path), passagem_melhor_crop)
                        _crop_path.with_suffix(".json").write_text(
                            json.dumps({
                                "horario":       datetime.now().isoformat(timespec="seconds"),
                                "duracao_fr":    passagem_frames_total,
                                "n_inferencias": passagem_inf_total,
                                "decisao":       _decisao_pass,
                                "clip_vencedor": _clip_opiniao,
                            }, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )

                        # Submeter ao VLM em thread (Change 3)
                        if usar_vlm:
                            vlm_em_andamento = True
                            threading.Thread(
                                target=_thread_vlm,
                                args=(
                                    passagem_melhor_crop.copy(),
                                    _clip_opiniao,
                                    list(nomes),
                                    _vlm_lock, _vlm_resultado,
                                    {
                                        "clip":    _clip_opiniao,
                                        "sim":     _sim_clip,
                                        "origem":  produto_via.get(_clip_opiniao, "clip")
                                                   if _clip_opiniao != "nenhum" else "clip",
                                        "decisao": _decisao_pass,
                                    },
                                ),
                                daemon=True,
                            ).start()

                    passagem_ativa          = False
                    passagem_frames_sem_det = 0
                    passagem_melhor_crop    = None
                    passagem_melhor_sharp   = 0.0

        # ---- Atualizar trilha (area / linha) ----
        linha_x_px = int(w * LINHA_X_FRAC)

        if modo == "auto":
            if direcao_ativa and inferencia_rodou and (
                    similaridade_atual >= limiar or via_ident == "detector"):
                if similaridade_atual > trilha_melhor_sim:
                    trilha_melhor_sim  = similaridade_atual
                    trilha_melhor_nome = melhor_palpite
                    trilha_melhor_via  = via_ident

            trilha_centroides.append(((x1 + x2) // 2, (y1 + y2) // 2))
            trilha_areas.append((x2 - x1) * (y2 - y1) / max(1, h * w))
            trilha_frames_sem_det = 0

            # Criterio linha (so quando direcao_ativa)
            if (direcao_ativa and modo_direcao == "linha"
                    and not trilha_cruzou and len(trilha_centroides) >= 2):
                cx_prev = trilha_centroides[-2][0]
                cx_curr = trilha_centroides[-1][0]
                if cx_prev < linha_x_px <= cx_curr:
                    direcao = "entrada"
                elif cx_prev >= linha_x_px > cx_curr:
                    direcao = "saida"
                else:
                    direcao = None
                if direcao:
                    feedback_texto, feedback_cor = _processar_evento(
                        direcao, trilha_melhor_nome, trilha_melhor_sim,
                        inventario, eventos, modo_evento="linha",
                        origem=trilha_melhor_via or "clip",
                    )
                    feedback_ate  = time.time() + FEEDBACK_DURACAO
                    trilha_cruzou = True

        elif deteccao_ativa and modo in ("fallback", "sem_fundo", "sem_objeto"):
            trilha_frames_sem_det += 1
            if trilha_frames_sem_det >= TRILHA_FRAMES_TIMEOUT:
                # Criterio area (so quando direcao_ativa)
                if direcao_ativa and modo_direcao == "area" and not trilha_cruzou:
                    direcao = _decidir_direcao_area(trilha_areas)
                    if direcao:
                        feedback_texto, feedback_cor = _processar_evento(
                            direcao, trilha_melhor_nome, trilha_melhor_sim,
                            inventario, eventos, modo_evento="area",
                            origem=trilha_melhor_via or "clip",
                        )
                        feedback_ate = time.time() + FEEDBACK_DURACAO
                (trilha_centroides, trilha_areas,
                 trilha_melhor_nome, trilha_melhor_sim,
                 trilha_frames_sem_det, trilha_cruzou) = _trilha_nova()
                trilha_melhor_via = ""

        # ---- Valores para HUD de area ----
        if direcao_ativa and modo_direcao == "area" and trilha_areas and modo == "auto":
            area_display = float(np.mean(trilha_areas[-AREA_JANELA_SUAVE:]))
            if len(trilha_areas) >= AREA_JANELA_SUAVE * 2:
                med_ini = float(np.mean(trilha_areas[:AREA_JANELA_SUAVE]))
                med_fim = float(np.mean(trilha_areas[-AREA_JANELA_SUAVE:]))
                razao_display = (med_fim / med_ini) if med_ini > 0 else None
            else:
                razao_display = None
        else:
            area_display  = None
            razao_display = None

        # ---- Processar resultado VLM (se disponivel) ----
        if usar_vlm:
            with _vlm_lock:
                if _vlm_resultado.get("pronto"):
                    _res             = dict(_vlm_resultado)
                    _vlm_resultado.clear()
                    vlm_em_andamento = False
                    _vlm_nome        = _res["vlm"]
                    _clip_log        = _res["clip"]
                    _concordam       = (_vlm_nome == _clip_log)
                    _dados_v         = _res["dados_evento"]
                    log_f.write(
                        f"  VLM clip={_clip_log}  vlm={_vlm_nome}"
                        f"  concord={_concordam}  decisao_clip={_dados_v['decisao']}\n"
                    )
                    nomes_set_vlm = set(nomes)
                    if _vlm_nome and _vlm_nome != "nenhum" and _vlm_nome in nomes_set_vlm:
                        feedback_texto, feedback_cor = _processar_evento(
                            "entrada", _vlm_nome, _dados_v["sim"],
                            inventario, eventos, modo_evento="reconhecimento",
                            origem=_dados_v["origem"] + "+vlm",
                        )
                        _recorte_ev = frame[qy1:qy2, qx1:qx2]
                        if _recorte_ev.size > 0:
                            n_log_evento += 1
                            _salvar_crop_log(_recorte_ev, sessao_crops_dir,
                                             "evento", n_log_evento, _vlm_nome)
                        feedback_ate = time.time() + FEEDBACK_DURACAO

        # ---- Construir reconhec_info ----
        reconhec_info = ""
        if usar_vlm and vlm_em_andamento:
            reconhec_info = "VLM: verificando..."
        elif not direcao_ativa and passagem_ativa:
            if passagem_acertos:
                por_prod_hud: dict = {}
                for prod, _, _ in passagem_acertos:
                    por_prod_hud[prod] = por_prod_hud.get(prod, 0) + 1
                lider, cnt = max(por_prod_hud.items(), key=lambda kv: kv[1])
                reconhec_info = f"passagem: {lider}  {cnt} acertos"
            else:
                reconhec_info = "passagem: aguardando..."

        # ---- Desenhar ----
        if direcao_ativa and modo_direcao == "linha":
            desenhar_linha_cruzamento(frame, linha_x_px)
        desenhar_trilha(frame, trilha_centroides)

        det_info = (f"{yolo_classe_atual}  {yolo_conf_atual:.2f}"
                    if detector == "yolo" and yolo_classe_atual else "")
        desenhar_frame(
            frame,
            guia=(x1, y1, x2, y2),
            nome=nome_atual,
            similaridade=similaridade_atual,
            melhor_palpite=melhor_palpite,
            limiar=limiar,
            n_produtos=len(nomes),
            modo=modo,
            com_enriquecido=com_enriquecido,
            direcao_flag=direcao_flag,
            modo_direcao=modo_direcao,
            detector=detector,
            det_info=det_info,
            reconhec_info=reconhec_info,
            via_ident=via_ident,
            decisao_atual=decisao_atual,
            margem_minima=margem_minima,
        )
        if direcao_ativa and modo_direcao == "area":
            desenhar_indicador_area(frame, area_display, razao_display)
        desenhar_inventario(frame, inventario)
        desenhar_feedback(frame, feedback_texto, feedback_cor, feedback_ate)

        cv2.imshow("Smart Fridge — Demo AP1", frame)
        tecla_raw = cv2.waitKeyEx(1)
        tecla     = tecla_raw & 0xFF

        # ---- Teclas ----
        if tecla in (ord("q"), ord("Q")):
            imprimir_resumo(inventario, eventos)
            if eventos or inventario:
                jp, cp = salvar_inventario(inventario, eventos, inventario_dir)
                print(f"\nInventario → {jp}")
                print(f"Eventos   → {cp}")
            break

        elif tecla in (ord("+"), ord("=")):
            limiar = min(1.0, round(limiar + 0.01, 2))
            print(f"Limiar ajustado: {limiar:.2f}")

        elif tecla == ord("-"):
            limiar = max(0.0, round(limiar - 0.01, 2))
            print(f"Limiar ajustado: {limiar:.2f}")

        elif tecla == ord("<"):
            margem_minima = max(0.0, round(margem_minima - 0.005, 3))
            print(f"Margem ajustada: {margem_minima:.3f}")

        elif tecla == ord(">"):
            margem_minima = min(0.50, round(margem_minima + 0.005, 3))
            print(f"Margem ajustada: {margem_minima:.3f}")

        elif tecla in (ord("b"), ord("B")):
            fundo_ref       = capturar_fundo(frame)
            caixa_suavizada = None
            (trilha_centroides, trilha_areas,
             trilha_melhor_nome, trilha_melhor_sim,
             trilha_frames_sem_det, trilha_cruzou) = _trilha_nova()
            trilha_melhor_via = ""
            if detector == "yolo":
                print("Fundo capturado. YOLO vai preferir objetos novos na cena.")
            else:
                print("Fundo capturado. Coloque o produto na cena.")

        elif tecla in (ord("d"), ord("D")):
            deteccao_ativa = not deteccao_ativa
            if deteccao_ativa:
                caixa_suavizada = None
                (trilha_centroides, trilha_areas,
                 trilha_melhor_nome, trilha_melhor_sim,
                 trilha_frames_sem_det, trilha_cruzou) = _trilha_nova()
                trilha_melhor_via = ""
                print(f"Deteccao: AUTO ({detector.upper()})")
            else:
                print("Deteccao: MANUAL (use setas para mover a caixa)")

        elif tecla in (ord("l"), ord("L")):
            if direcao_ativa:
                modo_direcao = "linha" if modo_direcao == "area" else "area"
                (trilha_centroides, trilha_areas,
                 trilha_melhor_nome, trilha_melhor_sim,
                 trilha_frames_sem_det, trilha_cruzou) = _trilha_nova()
                trilha_melhor_via = ""
                print(f"Criterio de direcao: {modo_direcao.upper()}")
            else:
                print("[L] sem efeito — use --direcao area|linha para ativar")

        elif tecla in (ord("a"), ord("A")):
            if melhor_palpite:
                feedback_texto, feedback_cor = _processar_evento(
                    "entrada", melhor_palpite, similaridade_atual,
                    inventario, eventos, modo_evento="manual",
                    origem=via_ident or "clip",
                )
                feedback_ate = time.time() + FEEDBACK_DURACAO
            else:
                print("[A] sem produto reconhecido no momento")

        elif tecla in (ord("s"), ord("S")):
            if melhor_palpite:
                feedback_texto, feedback_cor = _processar_evento(
                    "saida", melhor_palpite, similaridade_atual,
                    inventario, eventos, modo_evento="manual",
                    origem=via_ident or "clip",
                )
                feedback_ate = time.time() + FEEDBACK_DURACAO
            else:
                print("[S] sem produto reconhecido no momento")

        elif tecla in (ord("r"), ord("R")):
            inventario.clear()
            eventos.clear()
            reconhec_bloqueado.clear()
            passagem_ativa          = False
            passagem_frames_sem_det = 0
            passagem_acertos        = []
            feedback_texto = "Inventario zerado"
            feedback_cor   = COR_AMARELO
            feedback_ate   = time.time() + FEEDBACK_DURACAO
            print("Inventario, eventos e estado de reconhecimento zerados.")

        elif tecla in (ord("c"), ord("C")):
            sugestao = melhor_palpite if melhor_palpite else ""
            salvar_enriquecido(frame[qy1:qy2, qx1:qx2].copy(), enriquecido_dir, sugestao)

        elif tecla == 32:   # ESPACO
            salvar_recorte(frame[qy1:qy2, qx1:qx2].copy(), testes_dir)

        elif tecla_raw in KEY_LEFT:
            if deteccao_ativa:
                deteccao_ativa = False
                print("Controle manual ativado.")
            caixa_manual = mover_caixa(caixa_manual, -PASSO_MOVER, 0, h, w)

        elif tecla_raw in KEY_RIGHT:
            if deteccao_ativa:
                deteccao_ativa = False
                print("Controle manual ativado.")
            caixa_manual = mover_caixa(caixa_manual, PASSO_MOVER, 0, h, w)

        elif tecla_raw in KEY_UP:
            if deteccao_ativa:
                deteccao_ativa = False
                print("Controle manual ativado.")
            caixa_manual = mover_caixa(caixa_manual, 0, -PASSO_MOVER, h, w)

        elif tecla_raw in KEY_DOWN:
            if deteccao_ativa:
                deteccao_ativa = False
                print("Controle manual ativado.")
            caixa_manual = mover_caixa(caixa_manual, 0, PASSO_MOVER, h, w)

        elif tecla == ord("["):
            if deteccao_ativa:
                deteccao_ativa = False
                print("Controle manual ativado.")
            caixa_manual = redimensionar_caixa(caixa_manual, -PASSO_RESIZE, h, w)

        elif tecla == ord("]"):
            if deteccao_ativa:
                deteccao_ativa = False
                print("Controle manual ativado.")
            caixa_manual = redimensionar_caixa(caixa_manual, PASSO_RESIZE, h, w)

    cap.release()
    cv2.destroyAllWindows()
    _log_resumo(log_f, n_inf_total, n_inf_aceitas, n_inf_rej_limiar, n_inf_ambiguas, ambiguas_pares)
    log_f.close()
    print(f"Log de sessao : logs_demo/sessao_{sessao_ts}.txt")
    print(f"Recortes      : {sessao_crops_dir}  ({n_log_evento} arquivo(s))")


if __name__ == "__main__":
    main()
