"""
Comparaison RF vs CNN : métriques par classe, variabilité entre seeds, rappel par patient,
et diagnostic du rôle du rythme (RR) sur les battements S.

Usage :
    python -m src.compare --cnn cnn1d_sqrt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import CLASSES, load_meta, load_split
from src.evaluate import SEEDS, confusion
from src.features import build_features
from src.models import CNN1D
from src.train import add_rhythm, predict, to_tensor


def ms(agg, key, i=None):
    m, s = np.asarray(agg[key]["mean"]), np.asarray(agg[key]["std"])
    return (float(m), float(s)) if i is None else (float(m[i]), float(s[i]))


def fmt(t):
    return "  n/a  " if np.isnan(t[0]) else f"{t[0]:.3f}±{t[1]:.3f}"


def fig_confusions(rf_cm, cnn_cm, out, cnn_name):
    """Matrices de confusion RF et CNN côte à côte (test, % par ligne, moyenne sur les seeds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.eda import BLUE_RAMP, INK, style
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, cm, title in zip(axes, (rf_cm, cnn_cm), ("RF (morphologie + RR)", f"CNN 1D ({cnn_name})")):
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cmn, cmap=BLUE_RAMP, vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
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
    path = out / "figures" / f"08_confusion_rf_vs_{cnn_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("->", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cnn", default="cnn1d_sqrt")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = p.parse_args()

    rf = json.loads((args.results / "rf_baseline.json").read_text())
    cnn = json.loads((args.results / f"{args.cnn}.json").read_text())
    rf_t = {k: rf[k]["test"] for k in ("morph", "rr", "all")}
    cnn_t = cnn["agg"]["test"]

    fig_confusions(np.asarray(rf_t["all"]["cm"]["mean"]), np.asarray(cnn_t["cm"]["mean"]), args.results, args.cnn)

    print("=== 1. Métriques globales (test, moyenne ± écart-type sur les seeds) ===")
    print(f"{'modèle':>22} {'accuracy':>14} {'macro-F1 (N,S,V,F)':>20}")
    for name, a in [("RF morpho seule", rf_t["morph"]), ("RF RR seuls", rf_t["rr"]),
                    ("RF morpho + RR", rf_t["all"]), (f"CNN ({args.cnn})", cnn_t)]:
        print(f"{name:>22} {fmt(ms(a, 'accuracy')):>14} {fmt(ms(a, 'macro_f1_main')):>20}")

    print("\n=== 2. Par classe (test) : sensibilité | VPP | F1 ===")
    for metric in ("sensitivity", "ppv", "f1"):
        print(f"-- {metric}")
        print(f"{'':>6}" + "".join(f"{n:>16}" for n in ("RF morpho+RR", "CNN")))
        for i, c in enumerate(CLASSES[:4]):
            print(f"{c:>6}" + f"{fmt(ms(rf_t['all'], metric, i)):>16}" + f"{fmt(ms(cnn_t, metric, i)):>16}")

    print("\n=== 3. Variabilité entre seeds du CNN (test) ===")
    for s, r in zip(SEEDS, cnn["per_seed"]):
        t = r["test"]
        print(f"seed {s}: macro-F1 {t['macro_f1_main']:.3f} | acc {t['accuracy']:.3f} | "
              f"F1 N/S/V/F = " + "/".join(f"{x:.2f}" for x in t["f1"][:4]) + f" | epoch {r['best_epoch']}")

    # ---- prédictions du CNN (toutes seeds) et RF seed 0, pour les analyses par patient
    meta = load_meta(args.processed)
    data = add_rhythm({s: load_split(args.processed, s) for s in ("train", "test")})
    Xte, Rte, _ = to_tensor(data["test"])
    y, rec, rr = data["test"]["y"], data["test"]["rec"], data["test"]["rr"]
    preds = []
    for seed in SEEDS:
        m = CNN1D()
        m.load_state_dict(torch.load(args.results / "models" / f"{args.cnn}_seed{seed}.pt", map_location="cpu"))
        preds.append(predict(m, Xte, Rte, "cpu"))
    Xf = {s: build_features(data[s], meta["pre"], "all")[0] for s in data}
    rf_model = RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2,
                                      n_jobs=-1, random_state=0).fit(Xf["train"], data["train"]["y"])
    rf_pred = rf_model.predict(Xf["test"])

    print("\n=== 4. Matrice de confusion du CNN (test, % par ligne, moyenne des seeds) ===")
    cm = np.mean([confusion(y, p) for p in preds], axis=0)
    cmn = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
    print(f"{'':>6}" + "".join(f"{c:>8}" for c in CLASSES))
    for i, c in enumerate(CLASSES[:4]):
        print(f"{c:>6}" + "".join(f"{v:>8.1f}" for v in cmn[i]))

    print("\n=== 5. Rappel par patient : CNN (moyenne seeds) vs RF (seed 0) ===")
    for cls_i, cls in ((1, "S"), (2, "V"), (3, "F")):
        rows = []
        for r in np.unique(rec):
            mk = (rec == r) & (y == cls_i)
            if mk.sum() >= 20:
                cnn_rec = np.mean([(p[mk] == cls_i).mean() for p in preds])
                rows.append((r, int(mk.sum()), cnn_rec, float((rf_pred[mk] == cls_i).mean())))
        rows.sort(key=lambda t: -t[1])
        print(f"-- {cls} (records avec >= 20 battements) : record | n | CNN | RF")
        for r, n, a, b in rows[:8]:
            print(f"   {r:>5} {n:>5} {a:>6.3f} {b:>6.3f}")

    print("\n=== 6. Rôle du rythme : rappel de S selon que le battement est prématuré (RR préc./moy. < 0.85) ===")
    s_mask = y == 1
    prem = rr[:, 3] < 0.85
    for label, mk in (("prématuré", s_mask & prem), ("non prématuré", s_mask & ~prem)):
        cnn_r = np.mean([(p[mk] == 1).mean() for p in preds])
        print(f"   {label:>14}: n={int(mk.sum()):>5} | CNN {cnn_r:.3f} | RF {float((rf_pred[mk] == 1).mean()):.3f}")
    mk = (y == 0)
    print(f"   Faux positifs S parmi les N : CNN {np.mean([(p[mk] == 1).mean() for p in preds]):.4f} "
          f"| RF {float((rf_pred[mk] == 1).mean()):.4f}")


if __name__ == "__main__":
    main()
