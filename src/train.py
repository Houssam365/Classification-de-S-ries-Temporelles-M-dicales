"""
Entraînement des réseaux avec perte pondérée, répété sur plusieurs seeds (moyenne ± écart-type).

Usage :
    python -m src.train --model cnn1d                          # poids racine plafonnés, 5 seeds
    python -m src.train --model cnn1d --weights none           # sans pondération (ablation)
    python -m src.train --model cnn1d --seeds 0 --epochs 2     # test rapide
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_processing import CLASSES, load_meta, load_split
from src.evaluate import SEEDS, aggregate, format_report, summarize
from src.features import rhythm_relative
from src.models import CNN1D, CNNRhythm

MODELS = {"cnn1d": CNN1D, "cnn_rhythm": CNNRhythm}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y, mode, cap=None):
    """Poids par classe, relatifs à la classe majoritaire (poids 1).

    none : tous à 1 | sqrt : racine de (n_max / n_c) | inv : n_max / n_c ; plafonné à `cap`.
    """
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    ratio = counts.max() / np.maximum(counts, 1)
    w = {"none": np.ones_like(ratio), "sqrt": np.sqrt(ratio), "inv": ratio}[mode]
    cap = cap if cap is not None else {"none": 1.0, "sqrt": 10.0, "inv": 100.0}[mode]
    return np.minimum(w, cap).astype(np.float32)


def add_rhythm(data):
    """Ajoute data[split]["r"] : RR relatifs (log-ratios) standardisés avec les statistiques du train."""
    raw = {s: rhythm_relative(d["rr"]) for s, d in data.items()}
    mu, sd = raw["train"].mean(axis=0), raw["train"].std(axis=0)
    for s, d in data.items():
        d["r"] = ((raw[s] - mu) / sd).astype(np.float32)
    return data


def to_tensor(split):
    return (torch.from_numpy(split["X"]).unsqueeze(1), torch.from_numpy(split["r"]),
            torch.from_numpy(split["y"]))


@torch.no_grad()
def predict(model, X, R, device, batch=2048):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device)
        logits = model(xb, R[i:i + batch].to(device)) if model.uses_rhythm else model(xb)
        out.append(logits.argmax(1).cpu())
    return torch.cat(out).numpy()


def train_one(model_name, data, weights, seed, epochs, patience, lr, batch, device):
    set_seed(seed)
    Xtr, Rtr, ytr = to_tensor(data["train"])
    Xva, Rva, yva = to_tensor(data["val"])
    Xte, Rte, _ = to_tensor(data["test"])
    model = MODELS[model_name]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(Xtr, Rtr, ytr), batch_size=batch, shuffle=True, generator=gen)

    best, best_state, wait, hist = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, rb, yb in loader:
            xb, rb, yb = xb.to(device), rb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, rb) if model.uses_rhythm else model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        val = summarize(data["val"]["y"], predict(model, Xva, Rva, device))
        hist.append({"epoch": epoch + 1, "train_loss": total / len(Xtr), "val_macro_f1": val["macro_f1_main"],
                     "val_acc": val["accuracy"]})
        if val["macro_f1_main"] > best:          # sélection du modèle sur la validation uniquement
            best, wait = val["macro_f1_main"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    out = {"val": summarize(data["val"]["y"], predict(model, Xva, Rva, device)),
           "test": summarize(data["test"]["y"], predict(model, Xte, Rte, device)),
           "history": hist, "best_epoch": int(np.argmax([h["val_macro_f1"] for h in hist]) + 1)}
    return out, best_state


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), default="cnn1d")
    p.add_argument("--weights", choices=["none", "sqrt", "inv"], default="sqrt")
    p.add_argument("--cap", type=float, default=None, help="plafond des poids (défaut : 10 sqrt, 100 inv)")
    p.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = add_rhythm({s: load_split(args.processed, s) for s in ("train", "val", "test")})
    w = class_weights(data["train"]["y"], args.weights, args.cap)
    tag = f"{args.model}_{args.weights}"
    print(f"{tag} | device={device} | seeds={args.seeds} | poids {dict(zip(CLASSES, w.round(2)))}")
    (args.out / "models").mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in args.seeds:
        t0 = time.time()
        res, state = train_one(args.model, data, w, seed, args.epochs, args.patience, args.lr, args.batch, device)
        torch.save(state, args.out / "models" / f"{tag}_seed{seed}.pt")
        runs.append(res)
        print(f"seed {seed}: meilleure epoch {res['best_epoch']:>2} | val macro-F1 {res['val']['macro_f1_main']:.3f} "
              f"| test macro-F1 {res['test']['macro_f1_main']:.3f} | acc {res['test']['accuracy']:.3f} | {time.time() - t0:.0f}s")

    agg = {s: aggregate([r[s] for r in runs]) for s in ("val", "test")}
    print(f"\n=== {tag} : test, moyenne ± écart-type sur {len(runs)} seeds ===")
    print(format_report(agg["test"]))
    print(f"(val macro-F1 N,S,V,F : {float(agg['val']['macro_f1_main']['mean']):.3f} ± {float(agg['val']['macro_f1_main']['std']):.3f})")

    payload = {"args": {k: str(v) for k, v in vars(args).items()}, "weights": w.tolist(),
               "per_seed": runs, "agg": agg}
    (args.out / f"{tag}.json").write_text(json.dumps(to_jsonable(payload), indent=1))
    print(f"-> {args.out / (tag + '.json')}")


if __name__ == "__main__":
    main()
