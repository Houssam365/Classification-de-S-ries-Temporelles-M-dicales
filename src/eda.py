"""
Analyse exploratoire de la MIT-BIH : tableau descriptif, battements typiques par classe,
contexte rythmique, intervalles RR, spectrogrammes et validation de la fenêtre de segmentation.

Usage :
    python -m src.eda            # lit data/processed, écrit results/figures/*.png et results/eda_summary.csv
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram

from src.data_processing import (AAMI_MAP, CLASSES, FS, POST, PRE, load_meta, load_split,
                                 load_record, segment_record)

# Une couleur fixe par classe (jamais réaffectée) : slots 1 à 5 de la palette validée
COLORS = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
FULL_NAMES = {"N": "N : normal", "S": "S : supraventriculaire", "V": "V : ventriculaire",
              "F": "F : fusion", "Q": "Q : inconnu / stimulé"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
WIDE = 180                       # demi-fenêtre large (0,5 s) pour juger la fenêtre de segmentation
SEED = 0


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 110,
    })


def save(fig, out, name):
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out / name)


# ---------------------------------------------------------------- tableau descriptif
def descriptive_table(splits, out):
    rows = []
    for name, d in splits.items():
        n = len(d["y"])
        for i, c in enumerate(CLASSES):
            k = int((d["y"] == i).sum())
            rows.append({"split": name, "classe": c, "n": k, "pct": 100 * k / n})
    df = pd.DataFrame(rows)
    piv = df.pivot(index="classe", columns="split", values="n").reindex(CLASSES)[list(splits)]
    piv["total"] = piv.sum(axis=1)
    piv["% total"] = (100 * piv["total"] / piv["total"].sum()).round(2)
    piv["ratio vs N"] = (piv.loc["N", "total"] / piv["total"]).round(1)
    piv.to_csv(out / "eda_summary.csv")
    print("\nFréquence d'échantillonnage : %d Hz | fenêtre : %d samples (%.3f s) | canal MLII"
          % (FS, PRE + POST, (PRE + POST) / FS))
    print(piv.to_string())
    return piv


def fig_class_distribution(splits, out):
    fig, axes = plt.subplots(1, len(splits), figsize=(11, 3.4), sharey=True)
    for ax, (name, d) in zip(axes, splits.items()):
        counts = [(d["y"] == i).sum() for i in range(len(CLASSES))]
        bars = ax.bar(CLASSES, np.maximum(counts, 0.8), color=[COLORS[c] for c in CLASSES], width=0.62)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 2e5)
        ax.set_title(f"{name} ({len(d['y']):,} battements)".replace(",", " "))
        ax.grid(axis="x", visible=False)
        for b, k in zip(bars, counts):
            ax.annotate(f"{k:,}".replace(",", " "), (b.get_x() + b.get_width() / 2, max(k, 0.8)),
                        ha="center", va="bottom", fontsize=8.5, color=INK2, xytext=(0, 2),
                        textcoords="offset points")
    axes[0].set_ylabel("Nombre de battements (échelle log)")
    fig.suptitle("Déséquilibre des classes : N représente ~90 % des battements", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    save(fig, out, "01_distribution_classes.png")


# ---------------------------------------------------------------- battements typiques
def fig_mean_beats(raw_dir, train_records, out):
    """Battement moyen ± écart-type par classe sur une fenêtre large (±0,5 s), avec la fenêtre retenue."""
    d = {}
    parts = [segment_record(raw_dir, r, WIDE, WIDE) for r in train_records]
    X = np.concatenate([p["X"] for p in parts])
    y = np.concatenate([p["y"] for p in parts])
    t = np.arange(-WIDE, WIDE) / FS * 1000
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    for i, (ax, c) in enumerate(zip(axes, CLASSES)):
        Xc = X[y == i]
        ax.axvspan(-PRE / FS * 1000, POST / FS * 1000, color=GRID, alpha=0.55, lw=0)
        sel = rng.choice(len(Xc), size=min(25, len(Xc)), replace=False)
        for s in sel:
            ax.plot(t, Xc[s], color=COLORS[c], lw=0.5, alpha=0.18)
        m, sd = Xc.mean(0), Xc.std(0)
        ax.fill_between(t, m - sd, m + sd, color=COLORS[c], alpha=0.18, lw=0)
        ax.plot(t, m, color=COLORS[c], lw=2)
        ax.axvline(0, color=MUTED, lw=0.7, ls=":")
        ax.set_title(f"{FULL_NAMES[c]} (n={len(Xc):,})".replace(",", " "), fontsize=9.5)
        ax.set_xlabel("Temps relatif au pic R (ms)")
        d[c] = (m, sd)
    axes[0].set_ylabel("Amplitude normalisée")
    fig.suptitle("Battement moyen ± écart-type par classe (train). Zone grise : fenêtre retenue "
                 f"(-{PRE / FS * 1000:.0f} / +{POST / FS * 1000:.0f} ms)", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    save(fig, out, "02_battements_moyens_par_classe.png")


# ---------------------------------------------------------------- contexte rythmique
def fig_rhythm_strip(raw_dir, out, record="208", start_s=60, dur_s=10):
    sig, samples, symbols = load_record(raw_dir, record)
    a, b = int(start_s * FS), int((start_s + dur_s) * FS)
    t = np.arange(a, b) / FS
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.plot(t, sig[a:b], color=INK2, lw=0.8)
    seen = set()
    for s, sym in zip(samples, symbols):
        if a <= s < b:
            c = AAMI_MAP[sym]
            ax.scatter(s / FS, sig[s], s=34, color=COLORS[c], zorder=3, edgecolor=SURFACE, lw=1,
                       label=FULL_NAMES[c] if c not in seen else None)
            seen.add(c)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude normalisée")
    ax.set_title(f"Record {record}, {dur_s} s d'ECG annoté : les battements V (vert) ont un QRS plus large "
                 "et une onde T inversée")
    ax.legend(frameon=False, ncol=len(seen), loc="upper right", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "03_bande_ecg_annotee.png")


# ---------------------------------------------------------------- RR
def rr_stats(train, out):
    """Distribution du RR précédent / moyenne locale et du RR suivant / moyenne locale."""
    y, rr = train["y"], train["rr"]
    pre_ratio = rr[:, 3]
    post_ratio = rr[:, 1] / rr[:, 2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, vals, title in zip(axes, [pre_ratio, post_ratio],
                               ["RR précédent / RR moyen local", "RR suivant / RR moyen local"]):
        data = [vals[y == i] for i in range(len(CLASSES))]
        bp = ax.boxplot(data, orientation="vertical", patch_artist=True, widths=0.55, showfliers=False,
                        medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=MUTED, lw=0.8),
                        capprops=dict(color=MUTED, lw=0.8), boxprops=dict(lw=0))
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(COLORS[c])
            patch.set_alpha(0.85)
        ax.axhline(1, color=MUTED, lw=0.8, ls=":")
        ax.set_xticks(range(1, 6), CLASSES)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Rapport (1 = rythme régulier)")
    fig.suptitle("Les battements S et V arrivent plus tôt que prévu ; les V sont suivis d'une pause",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    save(fig, out, "04_rr_par_classe.png")

    rows = []
    for i, c in enumerate(CLASSES):
        m = y == i
        rows.append({"classe": c, "RR préc. médian (s)": np.median(rr[m, 0]),
                     "RR suiv. médian (s)": np.median(rr[m, 1]),
                     "préc./moy. médian": np.median(pre_ratio[m]),
                     "suiv./moy. médian": np.median(post_ratio[m]),
                     "FC médiane (bpm)": 60 / np.median(rr[m, 0])})
    df = pd.DataFrame(rows).set_index("classe").round(2)
    print("\nStatistiques RR par classe (train) :")
    print(df.to_string())
    return df


def window_overlap(splits):
    """Proportion de battements dont la fenêtre contient un autre pic R."""
    d = {k: np.concatenate([s[k] for s in splits.values()]) for k in ("y", "rr")}
    prev_in = d["rr"][:, 0] * FS < PRE
    next_in = d["rr"][:, 1] * FS < POST
    rows = []
    for i, c in enumerate(CLASSES):
        m = d["y"] == i
        rows.append({"classe": c, "R précédent dans la fenêtre (%)": 100 * prev_in[m].mean(),
                     "R suivant dans la fenêtre (%)": 100 * next_in[m].mean()})
    rows.append({"classe": "tous", "R précédent dans la fenêtre (%)": 100 * prev_in.mean(),
                 "R suivant dans la fenêtre (%)": 100 * next_in.mean()})
    df = pd.DataFrame(rows).set_index("classe").round(2)
    print(f"\nChevauchement de la fenêtre (-{PRE}/+{POST} samples) avec les battements voisins :")
    print(df.to_string())
    return df


# ---------------------------------------------------------------- fréquentiel
def fig_spectrograms(train, out, per_class=400):
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=True, layout="constrained")
    specs, vmax = {}, 0
    for i, c in enumerate(CLASSES):
        Xc = train["X"][train["y"] == i]
        sel = rng.choice(len(Xc), size=min(per_class, len(Xc)), replace=False)
        S = None
        for s in sel:
            f, tt, Sxx = spectrogram(Xc[s], fs=FS, nperseg=40, noverlap=36, nfft=128)
            S = Sxx if S is None else S + Sxx
        specs[c] = (f, tt, 10 * np.log10(S / len(sel) + 1e-12))
        vmax = max(vmax, specs[c][2].max())
    vmin = vmax - 50
    for ax, c in zip(axes, CLASSES):
        f, tt, S = specs[c]
        im = ax.pcolormesh((tt - PRE / FS) * 1000, f, S, cmap=BLUE_RAMP, vmin=vmin, vmax=vmax,
                           shading="auto")
        ax.set_ylim(0, 60)
        ax.set_title(FULL_NAMES[c], fontsize=9.5)
        ax.set_xlabel("Temps relatif au pic R (ms)")
        ax.grid(False)
    axes[0].set_ylabel("Fréquence (Hz)")
    cb = fig.colorbar(im, ax=axes, pad=0.01, fraction=0.015)
    cb.set_label("Puissance moyenne (dB)")
    fig.suptitle("Spectrogramme moyen d'un battement par classe (train)", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    save(fig, out, "05_spectrogrammes_par_classe.png")


def fig_psd(train, out):
    from scipy.signal import welch
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, c in enumerate(CLASSES):
        Xc = train["X"][train["y"] == i]
        Xc = Xc[np.random.default_rng(SEED).choice(len(Xc), size=min(3000, len(Xc)), replace=False)]
        f, P = welch(Xc, fs=FS, nperseg=128, axis=1)
        ax.plot(f, 10 * np.log10(P.mean(0)), color=COLORS[c], lw=1.8, label=FULL_NAMES[c])
    ax.set_xlim(0, 60)
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Densité spectrale moyenne (dB)")
    ax.set_title("Spectre moyen par classe : V et F (QRS plus large) ont moins d'énergie > 15 Hz")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "06_spectre_par_classe.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--raw", type=Path, default=Path("data/raw/mitdb"))
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    style()
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.processed)
    splits = {n: load_split(args.processed, n) for n in ("train", "val", "test")}

    descriptive_table(splits, args.out)
    fig_class_distribution(splits, fig_dir)
    fig_mean_beats(args.raw, meta["splits"]["train"], fig_dir)
    fig_rhythm_strip(args.raw, fig_dir)
    rr_stats(splits["train"], fig_dir)
    window_overlap(splits)
    fig_spectrograms(splits["train"], fig_dir)
    fig_psd(splits["train"], fig_dir)


if __name__ == "__main__":
    main()
