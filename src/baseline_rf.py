"""
Baseline de référence : features hand-crafted + Random Forest, avec ablation morphologie / RR / les deux,
répétée sur plusieurs seeds (moyenne ± écart-type).

Usage :
    python -m src.baseline_rf
    python -m src.baseline_rf --trees 300 --seeds 0 1 2 3 4
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import CLASSES, load_meta, load_split
from src.eda import BLUE_RAMP, GRID, INK, INK2, MUTED, SURFACE, style
from src.evaluate import SEEDS, aggregate, format_report, summarize
from src.features import build_features

KINDS = {"morph": "morphologie seule", "rr": "RR seuls", "all": "morphologie + RR"}


def run_config(kind, data, pre, seeds, trees):
    X = {s: build_features(d, pre, kind)[0] for s, d in data.items()}
    names = build_features(data["train"], pre, kind)[1]
    res = {"val": [], "test": []}
    importances, test_pred = [], []
    for seed in seeds:
        rf = RandomForestClassifier(n_estimators=trees, class_weight="balanced_subsample",
                                    min_samples_leaf=2, n_jobs=-1, random_state=seed)
        rf.fit(X["train"], data["train"]["y"])
        for s in ("val", "test"):
            pred = rf.predict(X[s])
            res[s].append(summarize(data[s]["y"], pred))
            if s == "test":
                test_pred.append(pred)
        importances.append(rf.feature_importances_)
    return res, np.mean(importances, axis=0), names


def fig_results(agg_all, importances, names, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    cm = agg_all["cm"]["mean"]
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    a1.imshow(cmn, cmap=BLUE_RAMP, vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            a1.text(j, i, f"{100 * cmn[i, j]:.0f}%", ha="center", va="center", fontsize=10,
                    color="white" if cmn[i, j] > 0.5 else INK)
    a1.set_xticks(range(5), CLASSES)
    a1.set_yticks(range(5), CLASSES)
    a1.set_xlabel("Classe prédite")
    a1.set_ylabel("Classe réelle")
    a1.grid(False)
    a1.set_title("Matrice de confusion (test, % par ligne)")

    top = np.argsort(importances)[::-1][:12][::-1]
    a2.barh([names[i] for i in top], importances[top], color="#2a78d6", height=0.62)
    a2.set_title("Importance des features (moyenne sur les seeds)")
    a2.set_xlabel("Importance (Gini)")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "07_rf_confusion_importances.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def to_jsonable(agg):
    return {k: {m: np.asarray(v).tolist() for m, v in d.items()} for k, d in agg.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = p.parse_args()

    style()
    meta = load_meta(args.processed)
    data = {s: load_split(args.processed, s) for s in ("train", "val", "test")}
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    summary, aggs = {}, {}
    for kind, label in KINDS.items():
        res, imp, names = run_config(kind, data, meta["pre"], args.seeds, args.trees)
        aggs[kind] = {s: aggregate(res[s]) for s in res}
        summary[kind] = {s: to_jsonable(aggs[kind][s]) for s in res}
        print(f"\n=== RF [{label}] : {len(args.seeds)} seeds, {args.trees} arbres ===")
        for s in ("val", "test"):
            print(f"-- {s} : accuracy {float(aggs[kind][s]['accuracy']['mean']):.3f} | macro-F1 (N,S,V,F) "
                  f"{float(aggs[kind][s]['macro_f1_main']['mean']):.3f} ± {float(aggs[kind][s]['macro_f1_main']['std']):.3f}")
        if kind == "all":
            print("\nDétail par classe (test) :")
            print(format_report(aggs[kind]["test"]))
            fig_results(aggs[kind]["test"], imp, names, args.out / "figures")
            summary["feature_importances"] = dict(zip(names, imp.tolist()))

    print("\n=== Ablation (test, macro-F1 N,S,V,F et F1 par classe, moyenne ± écart-type) ===")
    print(f"{'config':>20} {'macro-F1':>14} " + " ".join(f"{'F1 ' + c:>14}" for c in CLASSES[:4]))
    for kind, label in KINDS.items():
        t = aggs[kind]["test"]
        cells = " ".join(f"{t['f1']['mean'][i]:>7.3f}±{t['f1']['std'][i]:<5.3f}" for i in range(4))
        print(f"{label:>20} {float(t['macro_f1_main']['mean']):>7.3f}±{float(t['macro_f1_main']['std']):<5.3f} {cells}")

    (args.out / "rf_baseline.json").write_text(json.dumps(summary, indent=2))
    print("\n-> results/rf_baseline.json, results/figures/07_rf_confusion_importances.png")


if __name__ == "__main__":
    main()
