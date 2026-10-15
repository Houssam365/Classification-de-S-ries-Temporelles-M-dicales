"""
Interprétabilité : Grad-CAM temporel pour les CNN 1D (CNN1D, CNNRhythm : branche morphologie uniquement).

Usage :
    python -m src.interpretability --tag cnn1d_sqrt --seed 0
    python -m src.interpretability --tag cnn_rhythm_sqrt --seed 0
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data_processing import CLASSES, FS, load_meta, load_split
from src.eda import BLUE_RAMP, COLORS, INK, INK2, MUTED, style
from src.train import MODELS, add_rhythm, to_tensor

QRS_HALF = 25      # samples (~70 ms) autour du R, pour mesurer la part de l'attention portée au QRS


def gradcam(model, x, r, target):
    """Grad-CAM 1D sur la sortie de `model.features` : retourne (batch, longueur) dans [0, 1].

    Poids des canaux = gradient du score de la classe cible moyenné dans le temps ; la carte est la somme
    pondérée des canaux, rectifiée (ReLU), ré-étirée à la longueur du signal et normalisée par battement.
    """
    model.eval()
    feats = model.features(x)
    feats.retain_grad()
    z = feats.mean(dim=-1)
    if model.uses_rhythm:
        z = torch.cat([z, model.rhythm(r)], dim=1)
    logits = model.head(z)
    logits.gather(1, target[:, None]).sum().backward()
    weights = feats.grad.mean(dim=-1, keepdim=True)
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)[:, 0]
    cam = cam / cam.amax(dim=1, keepdim=True).clamp_min(1e-9)
    return cam.detach(), logits.detach().argmax(1)


def compute_all(model, X, R, y, batch=1024):
    cams, preds = [], []
    for i in range(0, len(X), batch):
        cam, pred = gradcam(model, X[i:i + batch], R[i:i + batch], y[i:i + batch])
        cams.append(cam)
        preds.append(pred)
    return torch.cat(cams).numpy(), torch.cat(preds).numpy()


def overlay(ax, t, beat, cam, color=INK2):
    ax.imshow(cam[None, :], aspect="auto", cmap=BLUE_RAMP, vmin=0, vmax=1, alpha=0.9,
              extent=[t[0], t[-1], beat.min() - 0.3, beat.max() + 0.3])
    ax.plot(t, beat, color=color, lw=1.4)
    ax.axvline(0, color=MUTED, lw=0.7, ls=":")
    ax.set_xlim(t[0], t[-1])
    ax.grid(False)


def fig_examples(X, y, pred, cam, t, out, tag, rng, n=3):
    fig, axes = plt.subplots(4, n, figsize=(3.6 * n, 8.4), sharex=True)
    for i, c in enumerate(CLASSES[:4]):
        ok = np.where((y == i) & (pred == i))[0]
        ko = np.where((y == i) & (pred != i))[0]
        pool = list(rng.choice(ok, min(len(ok), n), replace=False)) if len(ok) else []
        pool += list(rng.choice(ko, n - len(pool), replace=False)) if len(pool) < n and len(ko) else []
        for j in range(n):
            ax = axes[i, j]
            if j >= len(pool):
                ax.axis("off")
                continue
            k = pool[j]
            overlay(ax, t, X[k, 0], cam[k])
            good = pred[k] == y[k]
            ax.set_title(f"vrai {c} → prédit {CLASSES[pred[k]]}" + ("" if good else "  (erreur)"), fontsize=9.5,
                         color=INK if good else "#d03b3b")
            if j == 0:
                ax.set_ylabel(f"classe {c}")
    for ax in axes[-1]:
        ax.set_xlabel("Temps relatif au pic R (ms)")
    fig.suptitle(f"Grad-CAM ({tag}) : zones qui soutiennent la vraie classe (foncé = importance forte)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out / f"10_gradcam_exemples_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_mean(X, y, pred, cam, t, out, tag):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for i, (ax, c) in enumerate(zip(axes, CLASSES[:4])):
        m = y == i
        ax.plot(t, cam[m].mean(0), color=COLORS[c], lw=2)
        ax.fill_between(t, 0, cam[m].mean(0), color=COLORS[c], alpha=0.2, lw=0)
        ax2 = ax.twinx()                      # battement moyen en repère, échelle sans enjeu de lecture
        ax2.plot(t, X[m, 0].mean(0), color=INK2, lw=0.9, alpha=0.8)
        ax2.set_yticks([])
        ax2.spines[["top", "right"]].set_visible(False)
        ax.axvline(0, color=MUTED, lw=0.7, ls=":")
        ax.set_title(f"{c} (n={int(m.sum()):,})".replace(",", " "), fontsize=10)
        ax.set_xlabel("Temps relatif au pic R (ms)")
    axes[0].set_ylabel("Grad-CAM moyen (0-1)")
    fig.suptitle(f"Grad-CAM moyen par classe ({tag}) ; trait gris : battement moyen", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out / f"11_gradcam_moyen_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="cnn1d_sqrt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = p.parse_args()

    style()
    meta = load_meta(args.processed)
    data = add_rhythm({s: load_split(args.processed, s) for s in ("train", "test")})
    X, R, y = to_tensor(data["test"])
    model = MODELS[args.tag.rsplit("_", 1)[0]]()
    model.load_state_dict(torch.load(args.results / "models" / f"{args.tag}_seed{args.seed}.pt", map_location="cpu"))

    cam, pred = compute_all(model, X, R, y)          # cible = vraie classe
    y_np, Xn = y.numpy(), X.numpy()
    t = (np.arange(-meta["pre"], meta["post"])) / FS * 1000
    fig_dir = args.results / "figures"
    rng = np.random.default_rng(0)
    fig_examples(Xn, y_np, pred, cam, t, fig_dir, args.tag, rng)
    fig_mean(Xn, y_np, pred, cam, t, fig_dir, args.tag)

    pre = meta["pre"]
    in_qrs = np.zeros(cam.shape[1], bool)
    in_qrs[pre - QRS_HALF:pre + QRS_HALF] = True
    print(f"{args.tag} (seed {args.seed}) : part de l'attention Grad-CAM dans le QRS (±{QRS_HALF / FS * 1000:.0f} ms), "
          f"{in_qrs.mean():.0%} du signal")
    for i, c in enumerate(CLASSES[:4]):
        m = y_np == i
        share = cam[m][:, in_qrs].sum(1) / np.maximum(cam[m].sum(1), 1e-9)
        print(f"  {c}: {np.mean(share):.0%} (n={int(m.sum())})")
    print("->", fig_dir / f"10_gradcam_exemples_{args.tag}.png", "|", fig_dir / f"11_gradcam_moyen_{args.tag}.png")


if __name__ == "__main__":
    main()
