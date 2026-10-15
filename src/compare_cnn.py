"""
Comparaison du CNN initial et du CNN avec branche rythme (RF en référence), avec les mêmes seeds.

Usage :
    python -m src.compare_cnn --a cnn1d_sqrt --b cnn_rhythm_sqrt
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_processing import CLASSES, load_split
from src.eda import BLUE_RAMP, INK, style
from src.evaluate import SEEDS, confusion
from src.train import MODELS, add_rhythm, predict, to_tensor


def ms(agg, key, i=None):
    m, s = np.asarray(agg[key]["mean"]), np.asarray(agg[key]["std"])
    return (float(m), float(s)) if i is None else (float(m[i]), float(s[i]))


def fmt(t):
    return "   n/a   " if np.isnan(t[0]) else f"{t[0]:.3f}±{t[1]:.3f}"


def load_preds(tag, results, X, R):
    model_name = tag.rsplit("_", 1)[0]
    preds = []
    for seed in SEEDS:
        m = MODELS[model_name]()
        m.load_state_dict(torch.load(results / "models" / f"{tag}_seed{seed}.pt", map_location="cpu"))
        preds.append(predict(m, X, R, "cpu"))
    return preds


def fig_confusions(cms, titles, out, name):
    style()
    fig, axes = plt.subplots(1, len(cms), figsize=(6 * len(cms), 4.8))
    for ax, cm, title in zip(axes, cms, titles):
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cmn, cmap=BLUE_RAMP, vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{100 * cmn[i, j]:.1f}%\n({cm[i, j]:.0f})", ha="center", va="center",
                        fontsize=8.5, color="white" if cmn[i, j] > 0.5 else INK)
        ax.set_xticks(range(5), CLASSES)
        ax.set_yticks(range(5), CLASSES)
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Classe réelle")
        ax.grid(False)
        ax.set_title(title)
    fig.suptitle("Matrices de confusion sur le test (% par ligne, effectifs moyens entre parenthèses)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    path = out / "figures" / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("->", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="cnn1d_sqrt", help="CNN initial")
    p.add_argument("--b", default="cnn_rhythm_sqrt", help="CNN avec branche rythme")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = p.parse_args()

    rf = json.loads((args.results / "rf_baseline.json").read_text())["all"]["test"]
    A = json.loads((args.results / f"{args.a}.json").read_text())
    B = json.loads((args.results / f"{args.b}.json").read_text())
    models = {"RF (morpho+RR)": rf, "CNN initial": A["agg"]["test"], "CNN + rythme": B["agg"]["test"]}

    print("=== 1. Global (test, moyenne ± écart-type sur 5 seeds) ===")
    print(f"{'modèle':>16} {'accuracy':>14} {'macro-F1 (N,S,V,F)':>20}")
    for n, a in models.items():
        print(f"{n:>16} {fmt(ms(a, 'accuracy')):>14} {fmt(ms(a, 'macro_f1_main')):>20}")

    print("\n=== 2. Par classe (test) ===")
    for metric in ("sensitivity", "ppv", "f1"):
        print(f"-- {metric}\n{'':>6}" + "".join(f"{n:>17}" for n in models))
        for i, c in enumerate(CLASSES[:4]):
            print(f"{c:>6}" + "".join(f"{fmt(ms(a, metric, i)):>17}" for a in models.values()))

    print("\n=== 3. Seed par seed : macro-F1 (test) et F1 de S ===")
    deltas = []
    for s, ra, rb in zip(SEEDS, A["per_seed"], B["per_seed"]):
        ta, tb = ra["test"], rb["test"]
        deltas.append(tb["macro_f1_main"] - ta["macro_f1_main"])
        print(f"seed {s}: initial {ta['macro_f1_main']:.3f} -> rythme {tb['macro_f1_main']:.3f} "
              f"({deltas[-1]:+.3f}) | F1 S {ta['f1'][1]:.3f} -> {tb['f1'][1]:.3f} | "
              f"F1 V {ta['f1'][2]:.3f} -> {tb['f1'][2]:.3f}")
    print(f"gain moyen {np.mean(deltas):+.3f} ± {np.std(deltas):.3f} | seeds améliorées : "
          f"{sum(d > 0 for d in deltas)}/{len(deltas)}")

    # ---- prédictions sur le test pour les analyses fines
    data = add_rhythm({s: load_split(args.processed, s) for s in ("train", "test")})
    X, R, _ = to_tensor(data["test"])
    y, rec, rr = data["test"]["y"], data["test"]["rec"], data["test"]["rr"]
    pa, pb = load_preds(args.a, args.results, X, R), load_preds(args.b, args.results, X, R)

    print("\n=== 4. Fausses alertes sur les N (battements normaux prédits anormaux) ===")
    n_mask = y == 0
    for name, ps in (("CNN initial", pa), ("CNN + rythme", pb)):
        fa = np.mean([(p[n_mask] != 0).sum() for p in ps])
        print(f"{name:>14}: {fa:>7.0f} / {int(n_mask.sum())} ({100 * fa / n_mask.sum():.1f} %) | "
              f"N->S {np.mean([(p[n_mask] == 1).sum() for p in ps]):.0f}, "
              f"N->V {np.mean([(p[n_mask] == 2).sum() for p in ps]):.0f}, "
              f"N->F {np.mean([(p[n_mask] == 3).sum() for p in ps]):.0f}")

    print("\n=== 5. Rappel par patient (moyenne des seeds) : initial | rythme ===")
    for ci, cls in ((1, "S"), (2, "V"), (3, "F")):
        rows = []
        for r in np.unique(rec):
            mk = (rec == r) & (y == ci)
            if mk.sum() >= 20:
                rows.append((r, int(mk.sum()), np.mean([(p[mk] == ci).mean() for p in pa]),
                             np.mean([(p[mk] == ci).mean() for p in pb])))
        rows.sort(key=lambda t: -t[1])
        print(f"-- {cls} : record | n | initial | rythme")
        for r, n, a, b in rows[:7]:
            print(f"   {r:>5} {n:>5} {a:>7.3f} {b:>7.3f}")

    print("\n=== 6. Rôle du rythme pour S (rappel selon prématuré : RR préc./moy. < 0.85) ===")
    s_mask, prem = y == 1, rr[:, 3] < 0.85
    for label, mk in (("prématuré", s_mask & prem), ("non prématuré", s_mask & ~prem)):
        print(f"   {label:>14}: n={int(mk.sum()):>5} | initial {np.mean([(p[mk] == 1).mean() for p in pa]):.3f} "
              f"| rythme {np.mean([(p[mk] == 1).mean() for p in pb]):.3f}")

    cms = [np.mean([confusion(y, p) for p in ps], axis=0) for ps in (pa, pb)]
    fig_confusions(cms, ["CNN initial", "CNN + branche rythme"], args.results, "09_confusion_cnn_vs_rythme.png")
    print("\nMatrice CNN + rythme (effectifs moyens) :")
    for i, c in enumerate(CLASSES[:4]):
        print(f"{c:>4} " + " ".join(f"{v:>8.0f}" for v in cms[1][i]))


if __name__ == "__main__":
    main()
