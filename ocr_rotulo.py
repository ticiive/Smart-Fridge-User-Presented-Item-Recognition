"""
ocr_rotulo.py — OCR de rotulo via Apple Vision, com pre-processamento e 4 rotacoes.

Importado por demo_ao_vivo.py e processar_capturas.py para evitar duplicacao.
"""

from __future__ import annotations

import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Stop-words e utilitarios de texto
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    # pt
    "de", "da", "do", "das", "dos", "com", "sem", "por", "para", "em",
    "um", "uma", "uns", "umas", "que", "nao", "sim", "ate", "sua", "seu",
    "van", "les", "des", "aux", "une",
    # en
    "the", "and", "for", "with", "from", "are", "has", "not", "per",
    "san", "mix",
}


def normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def construir_indice_palavras(palavras: dict) -> dict:
    """
    Retorna {palavra_normalizada: nome_produto} apenas para palavras que:
      - tenham >= 3 letras
      - nao sejam stop-word
      - aparecam em EXATAMENTE UM produto do catalogo

    Imprime as palavras descartadas por serem compartilhadas entre produtos.
    """
    contagem: dict = {}
    for nome, kws in palavras.items():
        for kw in kws:
            kn = normalizar(kw)
            if len(kn) < 3 or kn in _STOP_WORDS:
                continue
            if kn not in contagem:
                contagem[kn] = []
            if nome not in contagem[kn]:
                contagem[kn].append(nome)

    indice: dict = {}
    compartilhadas = []
    for kw, prods in sorted(contagem.items()):
        if len(prods) == 1:
            indice[kw] = prods[0]
        else:
            compartilhadas.append(kw)

    if compartilhadas:
        print(f"  Palavras descartadas (compartilhadas entre produtos): "
              f"{', '.join(compartilhadas)}")
    print(f"  {len(indice)} palavra(s) unicas para matching OCR.")
    return indice


def match_palavras(texto: str, indice: dict) -> str:
    """
    indice: {palavra_normalizada: nome_produto} — saida de construir_indice_palavras.
    Conta votos de palavras unicas encontradas no texto OCR.
    """
    if not texto.strip() or not indice:
        return "nenhum"
    texto_norm = normalizar(texto)
    votos: dict = {}
    for kw, nome in indice.items():
        if kw in texto_norm:
            votos[nome] = votos.get(nome, 0) + 1
    if not votos:
        return "nenhum"
    return max(votos, key=lambda n: votos[n])


# ---------------------------------------------------------------------------
# OCR Apple Vision
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


def _preprocessar_para_ocr(img_path: Path) -> Optional[np.ndarray]:
    """
    Amplia 2x (INTER_CUBIC), converte para cinza, aplica CLAHE (clipLimit=2.0,
    tileGridSize=8x8) e unsharp mask.  Retorna array HxW uint8 ou None.
    """
    arr = cv2.imread(str(img_path))
    if arr is None:
        return None
    h, w = arr.shape[:2]
    arr  = cv2.resize(arr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    blur  = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


def _ocr_array(arr: np.ndarray) -> str:
    """Grava array em PNG temporario e roda Vision OCR nele."""
    try:
        import Vision
        from Foundation import NSURL
    except ImportError:
        return ""
    fd, tmp = tempfile.mkstemp(suffix=".png")
    try:
        os.close(fd)
        cv2.imwrite(tmp, arr)
        url     = NSURL.fileURLWithPath_(str(Path(tmp).resolve()))
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
        print(f"  [ocr] erro Vision: {exc}")
        return ""
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def ocr_imagem(img_path: Path) -> tuple[str, str, int]:
    """
    Pre-processa o recorte e testa 4 rotacoes (0, 90, 180, 270 graus).
    Escolhe a rotacao com mais caracteres alfanumericos.
    Retorna (texto_para_matching, texto_bruto, rotacao_vencedora).
    """
    if not checar_vision():
        return "", "", 0
    arr = _preprocessar_para_ocr(img_path)
    if arr is None:
        return "", "", 0
    candidatos = [
        (0,   arr),
        (90,  cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(arr, cv2.ROTATE_180)),
        (270, cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]
    melhor_graus = 0
    melhor_texto = ""
    melhor_n     = -1
    for graus, rot_arr in candidatos:
        texto = _ocr_array(rot_arr)
        n = sum(1 for c in texto if c.isalnum())
        if n > melhor_n:
            melhor_n     = n
            melhor_texto = texto
            melhor_graus = graus
    return melhor_texto, melhor_texto, melhor_graus
