"""
Avalia o reconhecimento em lote sobre a pasta testes/.

Formato dos arquivos de teste: <nome_do_produto>_<numero>.jpg
  - o sufixo _<numero> é removido para obter o rótulo verdadeiro
    (ex: leite_italac_2.jpg → leite_italac)
  - se o rótulo não estiver no catálogo, a imagem é tratada como "fora do
    catálogo" e o sistema deve rejeitar

Uso:
  python avaliar.py
  python avaliar.py --threshold 0.8
  python avaliar.py --sweep 0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90
  python avaliar.py --testes outra_pasta/ --saida resultado_exp1
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from recognizer import embed_image, get_device, load_index, load_model, match


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_true_label(filename: str) -> str:
    """'leite_italac_3.jpg' → 'leite_italac'"""
    return re.sub(r"_\d+$", "", Path(filename).stem)


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class ImageRecord:
    path: Path
    true_label: str
    in_catalog: bool
    emb: np.ndarray = field(repr=False)


@dataclass
class EvalResult:
    path: Path
    true_label: str
    in_catalog: bool
    prediction: Optional[str]
    similarity: float
    margin: float
    outcome: str
    # outcome:
    #   "correct"              — in_catalog, pred == true_label
    #   "wrong"                — in_catalog, pred != true_label (acima do limiar)
    #   "rejected_in"          — in_catalog, pred == None (abaixo do limiar)
    #   "correct_rejection"    — not in_catalog, pred == None  ✓
    #   "incorrect_assignment" — not in_catalog, pred != None  ✗


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def run_at_threshold(
    records: list[ImageRecord],
    catalog_names: list[str],
    catalog_embs: np.ndarray,
    threshold: float,
) -> list[EvalResult]:
    results = []
    for rec in records:
        pred, sim, margin = match(rec.emb, catalog_names, catalog_embs, threshold)
        if rec.in_catalog:
            if pred is None:
                outcome = "rejected_in"
            elif pred == rec.true_label:
                outcome = "correct"
            else:
                outcome = "wrong"
        else:
            outcome = "correct_rejection" if pred is None else "incorrect_assignment"
        results.append(EvalResult(
            path=rec.path,
            true_label=rec.true_label,
            in_catalog=rec.in_catalog,
            prediction=pred,
            similarity=sim,
            margin=margin,
            outcome=outcome,
        ))
    return results


def compute_metrics(results: list[EvalResult], threshold: float) -> dict:
    in_cat  = [r for r in results if r.in_catalog]
    out_cat = [r for r in results if not r.in_catalog]
    n_in, n_out = len(in_cat), len(out_cat)

    correct    = sum(1 for r in in_cat if r.outcome == "correct")
    wrong      = sum(1 for r in in_cat if r.outcome == "wrong")
    rej_in     = sum(1 for r in in_cat if r.outcome == "rejected_in")
    identified = correct + wrong  # acima do limiar

    cr = sum(1 for r in out_cat if r.outcome == "correct_rejection")
    ia = sum(1 for r in out_cat if r.outcome == "incorrect_assignment")

    return dict(
        threshold    = threshold,
        n_in         = n_in,
        n_out        = n_out,
        correct      = correct,
        wrong        = wrong,
        rej_in       = rej_in,
        # acurácia top-1: corretos / total no catálogo (penaliza rejeições)
        accuracy     = correct / n_in if n_in else 0.0,
        # precisão: dos que passaram o limiar, quantos acertaram
        precision    = correct / identified if identified else 0.0,
        # taxa de rejeição: imagens no catálogo que caíram abaixo do limiar
        rej_rate_in  = rej_in / n_in if n_in else 0.0,
        cr           = cr,
        ia           = ia,
        # taxa de rejeição correta: imagens fora do catálogo corretamente rejeitadas
        rej_rate_out = cr / n_out if n_out else 0.0,
    )


# ---------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------

def print_report(m: dict, results: list[EvalResult]):
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  RELATÓRIO — limiar: {m['threshold']:.4f}")
    print(sep)

    print(f"\n  No catálogo ({m['n_in']} imagens):")
    print(f"    Corretos          : {m['correct']:3d}  ({m['accuracy']*100:.1f}% acurácia top-1)")
    print(f"    Errados           : {m['wrong']:3d}")
    print(f"    Não reconhecidos  : {m['rej_in']:3d}  ({m['rej_rate_in']*100:.1f}% rejeitados)")
    if m["correct"] + m["wrong"] > 0:
        print(f"    Precisão (acima do limiar): {m['precision']*100:.1f}%")

    if m["n_out"]:
        print(f"\n  Fora do catálogo ({m['n_out']} imagens):")
        print(f"    Rejeições corretas  : {m['cr']:3d}  ({m['rej_rate_out']*100:.1f}%)")
        print(f"    Atribuições erradas : {m['ia']:3d}")

    errors = [r for r in results if r.outcome in ("wrong", "rejected_in", "incorrect_assignment")]
    if errors:
        print(f"\n  Erros ({len(errors)}):")
        print(f"  {'Arquivo':<32} {'Verdadeiro':<22} {'Previsto':<22} {'Sim':>6}")
        print(f"  {'-'*32} {'-'*22} {'-'*22} {'-'*6}")
        for r in errors:
            pred_str = r.prediction or "NAO_RECONHECIDO"
            print(f"  {r.path.name:<32} {r.true_label:<22} {pred_str:<22} {r.similarity:.4f}")
    else:
        print("\n  Nenhum erro!")


def save_results_csv(results: list[EvalResult], out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "arquivo", "rotulo_verdadeiro", "no_catalogo",
            "predicao", "similaridade", "margem", "resultado",
        ])
        w.writeheader()
        for r in results:
            w.writerow(dict(
                arquivo=r.path.name,
                rotulo_verdadeiro=r.true_label,
                no_catalogo=r.in_catalog,
                predicao=r.prediction or "NAO_RECONHECIDO",
                similaridade=f"{r.similarity:.4f}",
                margem=f"{r.margin:.4f}",
                resultado=r.outcome,
            ))
    print(f"  Resultados: {out_path}")


def save_sweep_csv(sweep: list[dict], out_path: Path):
    fields = [
        "threshold", "n_in", "n_out",
        "correct", "wrong", "rej_in",
        "accuracy", "precision", "rej_rate_in",
        "cr", "ia", "rej_rate_out",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in sweep:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in m.items()})
    print(f"  Varredura:  {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testes", default="testes/",
                        help="Pasta com imagens de teste (padrão: testes/)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--index", default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Substitui o limiar do config.yaml")
    parser.add_argument(
        "--sweep",
        default=None,
        metavar="T1,T2,...",
        help="Varre múltiplos limiares: ex. 0.5,0.6,0.7,0.8",
    )
    parser.add_argument("--saida", default=None,
                        help="Prefixo dos CSVs de saída (padrão: avaliacao_<timestamp>)")
    args = parser.parse_args()

    config = load_config(args.config)
    index_path = Path(args.index) if args.index else Path(config["index"]["path"])
    testes_dir = Path(args.testes)

    if not index_path.exists():
        sys.exit(f"Índice não encontrado: {index_path}\nExecute: python build_index.py")
    if not testes_dir.is_dir():
        sys.exit(f"Pasta de testes não encontrada: {testes_dir}")

    device = get_device()
    print(f"Dispositivo: {device}")
    model, preprocess = load_model(config, device)
    catalog_names, catalog_embs = load_index(index_path)
    catalog_set = set(catalog_names)
    print(f"Catálogo: {len(catalog_names)} produto(s)")

    test_images = sorted(
        p for p in testes_dir.iterdir() if p.suffix.lower() in SUPPORTED
    )
    if not test_images:
        sys.exit(f"Nenhuma imagem encontrada em {testes_dir}")

    print(f"Imagens de teste: {len(test_images)}  — calculando embeddings...")
    records: list[ImageRecord] = []
    for img_path in test_images:
        true_label = extract_true_label(img_path.name)
        records.append(ImageRecord(
            path=img_path,
            true_label=true_label,
            in_catalog=true_label in catalog_set,
            emb=embed_image(img_path, model, preprocess, device),
        ))

    n_in  = sum(1 for r in records if r.in_catalog)
    n_out = len(records) - n_in
    print(f"  No catálogo: {n_in}  |  Fora do catálogo: {n_out}")
    if n_out:
        fora = sorted({r.true_label for r in records if not r.in_catalog})
        print(f"  Rótulos fora: {', '.join(fora)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.saida or f"avaliacao_{ts}"
    Path("logs").mkdir(exist_ok=True)

    if args.sweep:
        thresholds = sorted(float(t.strip()) for t in args.sweep.split(","))
        sweep_data = []

        col = f"\n  {'Limiar':>6}  {'Acurácia':>9}  {'Precisão':>9}  {'Rej.In':>7}  {'Rej.Out':>9}"
        print(col)
        print("  " + "-" * (len(col) - 3))

        for thr in thresholds:
            res = run_at_threshold(records, catalog_names, catalog_embs, thr)
            m = compute_metrics(res, thr)
            sweep_data.append(m)
            print(
                f"  {thr:.3f}   "
                f"{m['accuracy']*100:7.1f}%  "
                f"{m['precision']*100:7.1f}%  "
                f"{m['rej_rate_in']*100:6.1f}%  "
                f"{m['rej_rate_out']*100:8.1f}%"
            )

        print()
        sweep_csv = Path("logs") / f"{prefix}_sweep.csv"
        save_sweep_csv(sweep_data, sweep_csv)

    else:
        thr = args.threshold if args.threshold is not None \
            else config["recognition"]["similarity_threshold"]
        results = run_at_threshold(records, catalog_names, catalog_embs, thr)
        m = compute_metrics(results, thr)
        print_report(m, results)
        results_csv = Path("logs") / f"{prefix}_resultados.csv"
        print()
        save_results_csv(results, results_csv)


if __name__ == "__main__":
    main()
