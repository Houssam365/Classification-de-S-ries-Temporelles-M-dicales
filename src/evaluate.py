"""
Métriques médicales par classe (sensibilité, spécificité, VPP, VPN, F1), matrice de confusion
et agrégation sur plusieurs seeds (moyenne ± écart-type).
"""
import warnings

import numpy as np

from src.data_processing import CLASSES

SEEDS = (0, 1, 2, 3, 4)       # seeds communes à tous les modèles : résultats rapportés en moyenne ± écart-type
N_MAIN = 4                    # N, S, V, F : Q (15 exemples) est trop rare pour une moyenne significative
METRICS = ["sensitivity", "specificity", "ppv", "npv", "f1"]


def confusion(y_true, y_pred, n_classes=len(CLASSES)) -> np.ndarray:
    """cm[i, j] = nombre de battements de vraie classe i prédits j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def class_metrics(cm: np.ndarray) -> dict:
    """Métriques en un-contre-tous pour chaque classe ; NaN quand elles ne sont pas définies."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    return {
        "sensitivity": _div(tp, tp + fn),            # rappel : part des vrais cas détectés
        "specificity": _div(tn, tn + fp),            # part des non-cas correctement écartés
        "ppv": _div(tp, tp + fp),                    # précision : part des alertes qui sont justes
        "npv": _div(tn, tn + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
    }


def summarize(y_true, y_pred) -> dict:
    cm = confusion(y_true, y_pred)
    m = class_metrics(cm)
    support = cm.sum(axis=1)
    present = support > 0
    main = present.copy()
    main[N_MAIN:] = False
    return {
        "cm": cm, **m,
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_f1": float(np.nanmean(m["f1"][present])),          # classes présentes dans y_true
        "macro_f1_main": float(np.nanmean(m["f1"][main])),        # N, S, V, F uniquement
    }


def aggregate(results: list) -> dict:
    """Moyenne et écart-type sur une liste de résultats `summarize` (un par seed)."""
    out = {}
    for key in ["cm", *METRICS, "accuracy", "macro_f1", "macro_f1_main"]:
        arr = np.stack([np.asarray(r[key], float) for r in results])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # classes sans exemple -> NaN voulu
            out[key] = {"mean": np.nanmean(arr, axis=0), "std": np.nanstd(arr, axis=0)}
    return out


def format_report(agg: dict) -> str:
    """Tableau texte : métriques par classe en moyenne ± écart-type."""
    lines = [f"{'classe':>6} " + " ".join(f"{m:>16}" for m in METRICS)]
    for i, c in enumerate(CLASSES):
        cells = []
        for m in METRICS:
            mu, sd = agg[m]["mean"][i], agg[m]["std"][i]
            cells.append(f"{'n/a':>16}" if np.isnan(mu) else f"{mu:>8.3f}±{sd:<6.3f} ")
        lines.append(f"{c:>6} " + " ".join(cells))
    for k, label in [("accuracy", "accuracy"), ("macro_f1_main", "macro-F1 (N,S,V,F)"),
                     ("macro_f1", "macro-F1 (5 cl.)")]:
        lines.append(f"{label:>20}: {float(agg[k]['mean']):.3f} ± {float(agg[k]['std']):.3f}")
    return "\n".join(lines)
