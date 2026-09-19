"""
Demo ao vivo — apresentação AP1.

Abre a webcam, exibe um quadrado guia no centro do frame e identifica o
produto dentro do quadrado a cada N frames usando embedding CLIP + cosseno.

Lê catalogo/*.jpg diretamente no startup; não depende de build_index.py.

Uso:
    python demo_ao_vivo.py
    python demo_ao_vivo.py --camera 1
    python demo_ao_vivo.py --listar-cameras
    python demo_ao_vivo.py --catalogo outra_pasta/

Teclas:
    q          Sair
    +  /  =    Aumentar limiar em 0.01
    -          Diminuir limiar em 0.01
    ESPAÇO     Salvar recorte atual em testes/<rótulo>_<n>.jpg
               (rótulo digitado no terminal; n é sequencial, nunca sobrescreve)
"""

import argparse
import sys
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
LIMIAR_INICIAL             = 0.75
AQUECIMENTO_FRAMES         = 30    # quadros descartados para o sensor estabilizar
LIMIAR_ESCURIDAO           = 5.0   # média de pixels abaixo disso → câmera provavelmente preta

MODELO_CLIP = "ViT-B-32"
PRETRAINED  = "laion2b_s34b_b79k"

EXTENSOES_SUPORTADAS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

COR_VERDE    = (80, 200, 80)
COR_VERMELHO = (60, 60, 220)
COR_CINZA    = (170, 170, 170)
COR_BRANCO   = (240, 240, 240)
COR_PRETO    = (0, 0, 0)
ALPHA_FUNDO  = 0.50   # opacidade do fundo dos rótulos de texto


# ---------------------------------------------------------------------------
# Device / modelo
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
# Catálogo
# ---------------------------------------------------------------------------

def carregar_catalogo(
    catalog_dir: Path, model, preprocess, device: str
) -> tuple[list[str], np.ndarray]:
    """
    Lê todas as imagens de catalog_dir, calcula embeddings CLIP normalizados
    e retorna (nomes, matrix_embeddings).
    """
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
        nomes.append(img_path.stem)
        embs.append(emb.cpu().float().numpy()[0])
        print(f"  [catalogo] {img_path.stem}")

    return nomes, np.array(embs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Embedding do recorte ao vivo
# ---------------------------------------------------------------------------

def embed_recorte(recorte_bgr: np.ndarray, model, preprocess, device: str) -> np.ndarray:
    """Converte recorte BGR (OpenCV) → embedding CLIP L2-normalizado."""
    pil    = Image.fromarray(cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB))
    tensor = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()[0]


# ---------------------------------------------------------------------------
# Primitivos de desenho
# ---------------------------------------------------------------------------

def _fundo_semitransparente(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    alpha: float,
):
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
    """Escreve texto com fundo escuro semitransparente para legibilidade."""
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
# Overlay principal
# ---------------------------------------------------------------------------

def desenhar_frame(
    frame: np.ndarray,
    guia: tuple[int, int, int, int],   # (x1, y1, x2, y2)
    nome: str,                          # "" = ainda sem inferência
    similaridade: float,
    melhor_palpite: str,
    limiar: float,
    n_produtos: int,
):
    h = frame.shape[0]
    x1, y1, x2, y2 = guia

    if not nome:
        # Antes da primeira inferência
        cv2.rectangle(frame, (x1, y1), (x2, y2), COR_CINZA, 2)
        texto_com_fundo(frame, "aguardando...", x1, y2 + 28, COR_CINZA, escala=0.65)
    else:
        reconhecido = similaridade >= limiar
        cor_guia    = COR_VERDE if reconhecido else COR_VERMELHO

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor_guia, 2)

        if reconhecido:
            linha1 = nome
            linha2 = f"sim: {similaridade:.3f}"
            cor1   = COR_VERDE
        else:
            linha1 = "NAO RECONHECIDO"
            linha2 = f"melhor palpite: {melhor_palpite}  ({similaridade:.3f})"
            cor1   = COR_VERMELHO

        texto_com_fundo(frame, linha1, x1, y2 + 30, cor1,    escala=0.85, espessura=2)
        texto_com_fundo(frame, linha2, x1, y2 + 58, COR_CINZA, escala=0.60, espessura=1)

    # Rodapé
    rodape = (
        f"Limiar: {limiar:.2f}  |  {n_produtos} produto(s)  |"
        f"  [+/-] ajustar  [SPC] salvar  [Q] sair"
    )
    texto_com_fundo(frame, rodape, 10, h - 10, COR_BRANCO, escala=0.50, espessura=1)


# ---------------------------------------------------------------------------
# Salvar recorte
# ---------------------------------------------------------------------------

def proximo_indice(testes_dir: Path, rotulo: str) -> int:
    """Conta arquivos testes/<rotulo>_*.jpg e retorna o próximo índice."""
    return len(list(testes_dir.glob(f"{rotulo}_*.jpg"))) + 1


def salvar_recorte(recorte_bgr: np.ndarray, testes_dir: Path):
    """Pede rótulo no terminal e salva recorte sem sobrescrever existentes."""
    print("\n[ESPAÇO] Rótulo do produto (sem espaços nem acentos):")
    rotulo = input("  > ").strip()
    if not rotulo:
        print("  Rótulo vazio — recorte descartado.")
        return
    n       = proximo_indice(testes_dir, rotulo)
    destino = testes_dir / f"{rotulo}_{n}.jpg"
    cv2.imwrite(str(destino), recorte_bgr)
    print(f"  Salvo: {destino}")


# ---------------------------------------------------------------------------
# Câmera — abertura, aquecimento e diagnóstico
# ---------------------------------------------------------------------------

def abrir_camera(indice: int) -> cv2.VideoCapture:
    """
    Tenta abrir com CAP_AVFOUNDATION (mais estável no macOS).
    Faz fallback para o backend padrão se falhar.
    """
    cap = cv2.VideoCapture(indice, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(indice)


def aquecer_camera(cap: cv2.VideoCapture, n: int = AQUECIMENTO_FRAMES):
    """Descarta n frames para o sensor estabilizar antes de usar a imagem."""
    for _ in range(n):
        cap.read()


def checar_imagem_escura(cap: cv2.VideoCapture, indice: int):
    """Lê um frame e avisa se a média de pixels indicar câmera preta."""
    ret, frame = cap.read()
    if not ret:
        return
    media = float(frame.mean())
    if media < LIMIAR_ESCURIDAO:
        print(
            f"\nAVISO: câmera {indice} parece não estar entregando imagem "
            f"(média de pixels = {media:.1f}).\n"
            f"Tente outro índice com --camera N  (ex.: --camera 1).\n"
        )


def listar_cameras():
    """Varre índices 0–3, imprime resolução e média de pixels de cada dispositivo."""
    print("Varrendo dispositivos de câmera (0 a 3)...\n")
    for i in range(4):
        cap = abrir_camera(i)
        if not cap.isOpened():
            print(f"  [{i}] não disponível")
            cap.release()
            continue
        aquecer_camera(cap, AQUECIMENTO_FRAMES)
        ret, frame = cap.read()
        if not ret:
            print(f"  [{i}] abriu mas não entregou frame após aquecimento")
            cap.release()
            continue
        h, w  = frame.shape[:2]
        media = float(frame.mean())
        aviso = "  ← parece preta" if media < LIMIAR_ESCURIDAO else ""
        print(f"  [{i}] {w}x{h}  média pixels: {media:5.1f}{aviso}")
        cap.release()
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogo", default=None,
                        help="Pasta do catálogo (padrão: config.yaml → download.catalog_dir)")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--camera",   type=int, default=0,
                        help="Índice do dispositivo de câmera (padrão: 0)")
    parser.add_argument("--listar-cameras", action="store_true",
                        help="Lista dispositivos disponíveis (0–3) com resolução e brilho médio, depois sai")
    args = parser.parse_args()

    if args.listar_cameras:
        listar_cameras()
        return

    config      = load_config(args.config)
    catalog_dir = Path(args.catalogo or config["download"]["catalog_dir"])
    testes_dir  = Path("testes")
    testes_dir.mkdir(exist_ok=True)

    if not catalog_dir.is_dir():
        sys.exit(f"Pasta de catálogo não encontrada: {catalog_dir}")

    imagens = [
        p for p in catalog_dir.iterdir()
        if p.suffix.lower() in EXTENSOES_SUPORTADAS
    ]
    if not imagens:
        sys.exit(
            f"Catálogo vazio: nenhuma imagem encontrada em '{catalog_dir}'.\n"
            "Execute primeiro:  python baixar_catalogo.py"
        )

    device = get_device()
    print(f"Dispositivo : {device}")
    print(f"Modelo      : {MODELO_CLIP} / {PRETRAINED}")
    print("Carregando modelo CLIP...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODELO_CLIP, pretrained=PRETRAINED, device=device
    )
    model.eval()

    print(f"Indexando {len(imagens)} imagem(ns) do catálogo...")
    nomes, embeddings = carregar_catalogo(catalog_dir, model, preprocess, device)
    print(f"Pronto — {len(nomes)} produto(s) carregado(s).\n")

    print(f"Abrindo câmera {args.camera}...")
    cap = abrir_camera(args.camera)
    if not cap.isOpened():
        sys.exit(
            f"Não foi possível abrir a câmera {args.camera}.\n"
            "Use --listar-cameras para ver os dispositivos disponíveis."
        )
    print(f"  Aquecendo ({AQUECIMENTO_FRAMES} frames)...")
    aquecer_camera(cap, AQUECIMENTO_FRAMES)
    checar_imagem_escura(cap, args.camera)

    limiar          = LIMIAR_INICIAL
    contador_frames = 0

    # Estado da última inferência
    nome_atual        = ""      # "" = inferência ainda não rodou
    similaridade_atual = 0.0
    melhor_palpite    = ""

    print("Janela aberta. Teclas: Q sair | +/- limiar | ESPAÇO salvar recorte")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Câmera encerrou o stream.")
            break

        h, w  = frame.shape[:2]
        lado  = min(h, w) // 2
        cx, cy = w // 2, h // 2
        x1 = cx - lado // 2
        y1 = cy - lado // 2
        x2 = x1 + lado
        y2 = y1 + lado

        # Inferência a cada N frames
        contador_frames += 1
        if contador_frames % INFERENCIA_A_CADA_N_FRAMES == 0:
            recorte = frame[y1:y2, x1:x2]
            query   = embed_recorte(recorte, model, preprocess, device)
            sims    = embeddings @ query          # cosine similarity
            idx     = int(np.argmax(sims))
            melhor_palpite    = nomes[idx]
            similaridade_atual = float(sims[idx])
            nome_atual = nomes[idx] if similaridade_atual >= limiar else "NAO RECONHECIDO"

        desenhar_frame(
            frame,
            guia=(x1, y1, x2, y2),
            nome=nome_atual,
            similaridade=similaridade_atual,
            melhor_palpite=melhor_palpite,
            limiar=limiar,
            n_produtos=len(nomes),
        )

        cv2.imshow("Smart Fridge — Demo AP1", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla in (ord("q"), ord("Q")):
            break
        elif tecla in (ord("+"), ord("=")):
            limiar = min(1.0, round(limiar + 0.01, 2))
            print(f"Limiar ajustado: {limiar:.2f}")
        elif tecla == ord("-"):
            limiar = max(0.0, round(limiar - 0.01, 2))
            print(f"Limiar ajustado: {limiar:.2f}")
        elif tecla == 32:   # ESPAÇO
            recorte_para_salvar = frame[y1:y2, x1:x2].copy()
            salvar_recorte(recorte_para_salvar, testes_dir)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
