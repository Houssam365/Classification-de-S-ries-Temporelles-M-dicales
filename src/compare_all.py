"""
Comparaison finale RF / CNN / CNN + rythme / CNN-LSTM : global, par classe, par patient, fausses alertes,
écarts appariés par seed, et analyse des poids d'attention du CNN-LSTM.

Usage :
    python -m src.compare_all
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.compare_cnn import fig_confusions, fmt, load_preds, ms
from src.data_processing import CLASSES, load_split
from src.eda import COLORS, INK2, MUTED, style
from src.evaluate import SEEDS, confusion
from src.train import add_rhythm, to_tensor


def fig_attention(attn_by_class, k, out):
    style()
    fig, ax = plt.subplots(figsize=(8, 4))
    pos = np.arange(-(k // 2), k // 2 + 1)
    for c in CLASSES[:4]:
        ax.plot(pos, attn_by_class[c], marker="o", ms=5, lw=1.8, color=COLORS[c], label=c)
    ax.axvline(0, color=MUTED, lw=0.7, ls=":")
    ax.set_xticks(pos, [f"{p:+d}" if p else "central" for p in pos])
    ax.set_xlabel("Position du battement dans la fenêtre (négatif = avant, positif = après)")
    ax.set_ylabel("Poids d'attention moyen")
    ax.set_title("CNN-LSTM : où se porte l'attention selon la classe réelle (test, moyenne des seeds)")
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9)
    fig.tight_layout()
    path = out / "figures" / "12_attention_par_classe.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("->", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = p.parse_args()
    R = args.results

    rf = json.loads((R / "rf_baseline.json").read_text())["all"]["test"]
    runs = {"CNN": "cnn1d_sqrt", "CNN+rythme": "cnn_rhythm_sqrt", "CNN-LSTM": "cnn_lstm_sqrt"}
    J = {n: json.loads((R / f"{t}.json").read_text()) for n, t in runs.items()}
    agg = {"RF": rf, **{n: j["agg"]["test"] for n, j in J.items()}}

    print("=== 1. Global (test, moyenne ± écart-type sur 5 seeds) ===")
    print(f"{'modèle':>12} {'accuracy':>14} {'macro-F1 (N,S,V,F)':>20}  | val macro-F1")
    for n, a in agg.items():
        v = J[n]["agg"]["val"] if n in J else None
        print(f"{n:>12} {fmt(ms(a, 'accuracy')):>14} {fmt(ms(a, 'macro_f1_main')):>20}  | "
              f"{fmt(ms(v, 'macro_f1_main')) if v else '-'}")

    print("\n=== 2. Par classe (test) ===")
    for metric in ("sensitivity", "ppv", "f1"):
        print(f"-- {metric}\n{'':>6}" + "".join(f"{n:>17}" for n in agg))
        for i, c in enumerate(CLASSES[:4]):
            print(f"{c:>6}" + "".join(f"{fmt(ms(a, metric, i)):>17}" for a in agg.values()))

    print("\n=== 3. Seed par seed : macro-F1 test (CNN / CNN+rythme / CNN-LSTM) ===")
    d_lstm_rhy = []
    for s, ra, rb, rc in zip(SEEDS, *(J[n]["per_seed"] for n in runs)):
        a, b, c = (r["test"]["macro_f1_main"] for r in (ra, rb, rc))
        d_lstm_rhy.append(c - b)
        print(f"seed {s}: {a:.3f} / {b:.3f} / {c:.3f} | CNN-LSTM - CNN+rythme = {c - b:+.3f} | "
              f"F1 S {rb['test']['f1'][1]:.2f}->{rc['test']['f1'][1]:.2f} | F1 V {rb['test']['f1'][2]:.2f}->{rc['test']['f1'][2]:.2f}")
    print(f"CNN-LSTM - CNN+rythme : {np.mean(d_lstm_rhy):+.3f} ± {np.std(d_lstm_rhy):.3f} | "
          f"seeds où CNN-LSTM >= CNN+rythme : {sum(d >= 0 for d in d_lstm_rhy)}/5")

    # ---- prédictions test
    data = add_rhythm({s: load_split(args.processed, s) for s in ("train", "test")})
    X, Rh, _ = to_tensor(data["test"])
    y, rec, rr = data["test"]["y"], data["test"]["rec"], data["test"]["rr"]
    preds = {"CNN": load_preds("cnn1d_sqrt", R, X, Rh), "CNN+rythme": load_preds("cnn_rhythm_sqrt", R, X, Rh),
             "CNN-LSTM": [np.load(R / "preds" / f"cnn_lstm_sqrt_seed{s}.npy") for s in SEEDS]}
    for n, ps in preds.items():       # cohérence avec les métriques enregistrées
        assert abs(np.mean([(p == y).mean() for p in ps]) - float(J[n]["agg"]["test"]["accuracy"]["mean"])) < 1e-6, n

    print("\n=== 4. Fausses alertes sur les N ===")
    nm = y == 0
    for n, ps in preds.items():
        fa = np.mean([(p[nm] != 0).sum() for p in ps])
        print(f"{n:>12}: {fa:>6.0f} / {int(nm.sum())} ({100 * fa / nm.sum():.1f} %) | "
              f"N->S {np.mean([(p[nm] == 1).sum() for p in ps]):.0f}, N->V {np.mean([(p[nm] == 2).sum() for p in ps]):.0f}, "
              f"N->F {np.mean([(p[nm] == 3).sum() for p in ps]):.0f}")

    print("\n=== 5. Rappel par patient (moyenne des seeds) : CNN | CNN+rythme | CNN-LSTM ===")
    for ci, cls in ((1, "S"), (2, "V"), (3, "F")):
        rows = []
        for r in np.unique(rec):
            mk = (rec == r) & (y == ci)
            if mk.sum() >= 20:
                rows.append((r, int(mk.sum()), *[np.mean([(p[mk] == ci).mean() for p in ps]) for ps in preds.values()]))
        rows.sort(key=lambda t: -t[1])
        print(f"-- {cls} : record | n | CNN | CNN+rythme | CNN-LSTM")
        for r, n, a, b, c in rows[:7]:
            print(f"   {r:>5} {n:>5} {a:>7.3f} {b:>9.3f} {c:>9.3f}")

    print("\n=== 6. Rappel de S selon prématuré (RR préc./moy. < 0.85) ===")
    sm, prem = y == 1, rr[:, 3] < 0.85
    for label, mk in (("prématuré", sm & prem), ("non prématuré", sm & ~prem)):
        print(f"   {label:>14}: n={int(mk.sum()):>5} | " + " | ".join(
            f"{n} {np.mean([(p[mk] == 1).mean() for p in ps]):.3f}" for n, ps in preds.items()))

    cms = [np.mean([confusion(y, p) for p in ps], axis=0) for ps in preds.values()]
    fig_confusions(cms, list(preds), R, "13_confusion_cnn_rythme_lstm.png")
    print("\nMatrice CNN-LSTM (effectifs moyens) :")
    for i, c in enumerate(CLASSES[:4]):
        print(f"{c:>4} " + " ".join(f"{v:>8.0f}" for v in cms[2][i]))

    print("\n=== 7. Poids d'attention du CNN-LSTM (test, moyenne des seeds, par classe réelle) ===")
    attn = np.mean([np.load(R / "attn" / f"cnn_lstm_sqrt_seed{s}.npy") for s in SEEDS], axis=0)
    k = attn.shape[1]
    by_class = {c: attn[y == i].mean(0) for i, c in enumerate(CLASSES[:4])}
    print(f"{'':>4}" + "".join(f"{p:>8}" for p in [f"{q:+d}" if q else "centre" for q in range(-(k // 2), k // 2 + 1)]))
    for c, a in by_class.items():
        print(f"{c:>4}" + "".join(f"{v:>8.3f}" for v in a))
    fig_attention(by_class, k, R)


if __name__ == "__main__":
    main()
