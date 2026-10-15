"""
Entraînement du CNN-LSTM avec attention sur des séquences de battements, même protocole que src/train.py
(perte pondérée, sélection sur la validation, plusieurs seeds, moyenne ± écart-type).

Usage :
    python -m src.train_seq                         # K=9 battements, poids racine plafonnés, 5 seeds
    python -m src.train_seq --seeds 0 --epochs 2    # test rapide
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data_processing import CLASSES, load_split
from src.evaluate import SEEDS, aggregate, format_report, summarize
from src.models import CNNLSTMAttention
from src.sequences import build_sequences
from src.train import class_weights, set_seed, to_jsonable


def gather(seq, starts, device):
    """Blocs de T battements démarrant à `starts` : tenseurs (B, T, L), (B, T), labels et indices d'origine des centres."""
    t, h = seq["t"], seq["k"] // 2
    idx = starts[:, None] + np.arange(t)
    x = torch.from_numpy(seq["X"][idx]).to(device)
    lr = torch.from_numpy(seq["lr"][idx]).to(device)
    centers = idx[:, h:t - h]
    return x, lr, torch.from_numpy(seq["y"][centers].ravel()).to(device), seq["orig"][centers].ravel()


@torch.no_grad()
def predict_seq(model, seq, device, batch_chunks=32):
    """Prédictions et poids d'attention par battement, dans l'ordre d'origine du split."""
    model.eval()
    preds = np.full(seq["n"], -1, np.int64)
    attn = np.zeros((seq["n"], seq["k"]), np.float32)
    starts = seq["starts"]
    for i in range(0, len(starts), batch_chunks):
        x, lr, _, orig = gather(seq, starts[i:i + batch_chunks], device)
        logits, alpha = model(x, lr, return_attn=True)
        real = orig >= 0
        preds[orig[real]] = logits.argmax(1).cpu().numpy()[real]
        attn[orig[real]] = alpha.cpu().numpy()[real]
    assert (preds >= 0).all(), "des battements n'ont pas été prédits"
    return preds, attn


def train_one(seqs, labels, weights, seed, args, device):
    set_seed(seed)
    model = CNNLSTMAttention(k=args.k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device), ignore_index=-1)
    rng = np.random.default_rng(seed)
    train_starts = seqs["train"]["starts"]

    best, best_state, wait, hist = -1.0, None, 0, []
    for epoch in range(args.epochs):
        model.train()
        total, n_seen = 0.0, 0
        order = rng.permutation(len(train_starts))
        for i in range(0, len(order), args.batch_chunks):
            x, lr, y, _ = gather(seqs["train"], train_starts[order[i:i + args.batch_chunks]], device)
            opt.zero_grad()
            loss = loss_fn(model(x, lr), y)
            loss.backward()
            opt.step()
            n_real = int((y >= 0).sum())
            total, n_seen = total + loss.item() * n_real, n_seen + n_real
        pred_val, _ = predict_seq(model, seqs["val"], device)
        val = summarize(labels["val"], pred_val)
        hist.append({"epoch": epoch + 1, "train_loss": total / n_seen, "val_macro_f1": val["macro_f1_main"],
                     "val_acc": val["accuracy"]})
        if val["macro_f1_main"] > best:
            best, wait = val["macro_f1_main"], 0
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    pv, _ = predict_seq(model, seqs["val"], device)
    pt, attn = predict_seq(model, seqs["test"], device)
    out = {"val": summarize(labels["val"], pv), "test": summarize(labels["test"], pt), "history": hist,
           "best_epoch": int(np.argmax([h["val_macro_f1"] for h in hist]) + 1)}
    return out, best_state, pt, attn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", choices=["none", "sqrt", "inv"], default="sqrt")
    p.add_argument("--k", type=int, default=9, help="battements par séquence (impair)")
    p.add_argument("--t", type=int, default=40, help="taille des blocs encodés d'un coup")
    p.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-chunks", type=int, default=8, help="blocs par lot (8 x 32 centres ~ 256 battements)")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {s: load_split(args.processed, s) for s in ("train", "val", "test")}
    seqs = {s: build_sequences(d, args.k, args.t) for s, d in data.items()}
    labels = {s: d["y"] for s, d in data.items()}
    w = class_weights(data["train"]["y"], args.weights)
    tag = f"cnn_lstm_{args.weights}"
    print(f"{tag} | K={args.k}, blocs de {args.t} | device={device} | seeds={args.seeds} | "
          f"poids {dict(zip(CLASSES, w.round(2)))}")
    for d in ("models", "preds", "attn"):
        (args.out / d).mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in args.seeds:
        t0 = time.time()
        res, state, pred_test, attn = train_one(seqs, labels, w, seed, args, device)
        torch.save(state, args.out / "models" / f"{tag}_seed{seed}.pt")
        np.save(args.out / "preds" / f"{tag}_seed{seed}.npy", pred_test)
        np.save(args.out / "attn" / f"{tag}_seed{seed}.npy", attn)
        runs.append(res)
        print(f"seed {seed}: meilleure epoch {res['best_epoch']:>2} | val macro-F1 {res['val']['macro_f1_main']:.3f} "
              f"| test macro-F1 {res['test']['macro_f1_main']:.3f} | acc {res['test']['accuracy']:.3f} | "
              f"{time.time() - t0:.0f}s ({(time.time() - t0) / len(res['history']):.0f}s/epoch)")

    agg = {s: aggregate([r[s] for r in runs]) for s in ("val", "test")}
    print(f"\n=== {tag} : test, moyenne ± écart-type sur {len(runs)} seeds ===")
    print(format_report(agg["test"]))
    print(f"(val macro-F1 N,S,V,F : {float(agg['val']['macro_f1_main']['mean']):.3f} ± "
          f"{float(agg['val']['macro_f1_main']['std']):.3f})")
    payload = {"args": {k_: str(v) for k_, v in vars(args).items()}, "weights": w.tolist(),
               "per_seed": runs, "agg": agg}
    (args.out / f"{tag}.json").write_text(json.dumps(to_jsonable(payload), indent=1))
    print(f"-> {args.out / (tag + '.json')}")


if __name__ == "__main__":
    main()
