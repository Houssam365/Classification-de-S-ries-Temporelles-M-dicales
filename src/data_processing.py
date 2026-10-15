"""
Prétraitement de la MIT-BIH Arrhythmia Database : segmentation en battements,
filtrage, normalisation par enregistrement, features RR et split par patient.

Usage :
    python -m src.data_processing --summary            # effectifs par record et par classe
    python -m src.data_processing                      # construit data/processed/{train,val,test}.npz
    python -m src.data_processing --pre 90 --post 126  # taille de fenêtre personnalisée
"""
import argparse
import json
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

FS = 360                       # Hz
PRE, POST = 90, 126            # samples avant/après le pic R (0,25 s / 0,35 s) -> 216 samples
BAND = (0.5, 40.0)             # Hz, passe-bande
LEAD = "MLII"
RR_LOCAL_N = 10                # nombre d'intervalles RR pour la moyenne locale
RR_MAX = 6.0                   # s ; au-delà c'est un trou d'annotation (ex. flutter du record 207), pas un rythme

# Regroupement AAMI EC57 : symbole MIT-BIH -> classe
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Records à rythme entraîné (stimulateur), exclus (norme AAMI)
EXCLUDED = ["102", "104", "107", "217"]

# Split inter-patients de de Chazal et al. (2004)
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
# Validation : sous-ensemble de DS1 choisi à la main d'après `--summary` pour contenir S, V et F
# sans priver le train des records dominants (208 pour F, 209 pour S).
VAL_RECORDS = ["118", "124", "205", "207", "223"]


def load_record(raw_dir: Path, name: str):
    """Retourne (signal MLII filtré et normalisé, positions R, symboles) d'un enregistrement."""
    path = str(Path(raw_dir) / name)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    # Sélection du canal par son nom (MLII n'est pas toujours le canal 0, ex. record 114)
    sig = rec.p_signal[:, rec.sig_name.index(LEAD)].astype(np.float64)

    sos = butter(4, BAND, btype="bandpass", fs=FS, output="sos")
    sig = sosfiltfilt(sos, sig)
    # Normalisation par record : calculée sans labels, sur un record appartenant à un seul split
    sig = (sig - np.median(sig)) / sig.std()

    keep = np.array([s in AAMI_MAP for s in ann.symbol])
    samples = ann.sample[keep]
    symbols = np.array(ann.symbol)[keep]
    return sig.astype(np.float32), samples, symbols


def rr_features(samples: np.ndarray) -> np.ndarray:
    """Features RR par battement : [RR précédent, RR suivant, moyenne locale, RR précédent / moyenne].

    Valeurs en secondes. La moyenne locale est causale (intervalles précédents uniquement).
    """
    n = len(samples)
    rr = np.minimum(np.diff(samples) / FS, RR_MAX)   # n-1 intervalles, plafonnés
    if n < 2:
        return np.zeros((n, 4), dtype=np.float32)
    pre = np.concatenate([[rr[0]], rr])              # 1er battement : pas de précédent -> on prend le suivant
    post = np.concatenate([rr, [rr[-1]]])            # dernier battement : pas de suivant -> on prend le précédent
    csum = np.concatenate([[0.0], np.cumsum(pre)])
    idx = np.arange(1, n + 1)
    lo = np.maximum(idx - RR_LOCAL_N, 0)
    local = (csum[idx] - csum[lo]) / (idx - lo)
    return np.stack([pre, post, local, pre / local], axis=1).astype(np.float32)


def segment_record(raw_dir, name, pre=PRE, post=POST):
    """Découpe un enregistrement en fenêtres centrées sur le pic R."""
    sig, samples, symbols = load_record(raw_dir, name)
    rr = rr_features(samples)                        # calculé avant d'écarter les battements de bord

    valid = (samples - pre >= 0) & (samples + post <= len(sig))
    samples, symbols, rr = samples[valid], symbols[valid], rr[valid]

    X = np.stack([sig[s - pre:s + post] for s in samples]) if len(samples) else np.empty((0, pre + post), np.float32)
    y = np.array([CLASS_TO_ID[AAMI_MAP[s]] for s in symbols], dtype=np.int64)
    return {
        "X": X.astype(np.float32), "y": y, "rr": rr,
        "rec": np.full(len(samples), name), "r_idx": samples.astype(np.int64), "symbol": symbols,
    }


def build_split(raw_dir, records, pre=PRE, post=POST):
    parts = [segment_record(raw_dir, r, pre, post) for r in records]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def class_counts(y) -> dict:
    return {c: int((y == i).sum()) for c, i in CLASS_TO_ID.items()}


def summary(raw_dir, pre=PRE, post=POST):
    """Effectifs par record et par classe (utile pour choisir la validation)."""
    records = DS1 + DS2
    print(f"{'record':>6} {'split':>5} " + " ".join(f"{c:>6}" for c in CLASSES) + f" {'total':>7}")
    totals = {"DS1": np.zeros(len(CLASSES), int), "DS2": np.zeros(len(CLASSES), int)}
    for r in records:
        d = segment_record(raw_dir, r, pre, post)
        cnt = np.bincount(d["y"], minlength=len(CLASSES))
        split = "DS1" if r in DS1 else "DS2"
        totals[split] += cnt
        print(f"{r:>6} {split:>5} " + " ".join(f"{c:>6}" for c in cnt) + f" {cnt.sum():>7}")
    for split, cnt in totals.items():
        print(f"{'TOTAL':>6} {split:>5} " + " ".join(f"{c:>6}" for c in cnt) + f" {cnt.sum():>7}")


def make_splits():
    """Retourne {train, val, test} -> liste de records, en vérifiant l'absence de fuite entre patients."""
    train = [r for r in DS1 if r not in VAL_RECORDS]
    splits = {"train": train, "val": list(VAL_RECORDS), "test": list(DS2)}
    assert set(VAL_RECORDS) <= set(DS1), "la validation doit être un sous-ensemble de DS1"
    assert not set(DS1) & set(DS2), "DS1 et DS2 se chevauchent"
    assert not set(EXCLUDED) & (set(DS1) | set(DS2)), "un record exclu est utilisé"
    seen = [r for v in splits.values() for r in v]
    assert len(seen) == len(set(seen)), "un record apparaît dans deux splits"
    return splits


def process_all(raw_dir, out_dir, pre=PRE, post=POST):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = make_splits()
    data = {}
    for name, records in splits.items():
        data[name] = build_split(raw_dir, records, pre, post)
        np.savez_compressed(out_dir / f"{name}.npz", **data[name])
        print(f"{name:>5}: {len(records):>2} records, {len(data[name]['y']):>6} battements, "
              f"{class_counts(data[name]['y'])}")

    # Scaler des features RR : ajusté sur le train uniquement
    rr_train = data["train"]["rr"]
    scaler = {"mean": rr_train.mean(axis=0).tolist(), "std": rr_train.std(axis=0).tolist()}
    meta = {"fs": FS, "pre": pre, "post": post, "band": BAND, "lead": LEAD,
            "classes": CLASSES, "splits": splits, "rr_scaler": scaler}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_split(processed_dir, name: str) -> dict:
    with np.load(Path(processed_dir) / f"{name}.npz", allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def load_meta(processed_dir) -> dict:
    return json.loads((Path(processed_dir) / "meta.json").read_text())


def main():
    p = argparse.ArgumentParser(description="Prétraitement MIT-BIH")
    p.add_argument("--raw", type=Path, default=Path("data/raw/mitdb"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--pre", type=int, default=PRE)
    p.add_argument("--post", type=int, default=POST)
    p.add_argument("--summary", action="store_true", help="affiche les effectifs par record et par classe")
    args = p.parse_args()

    if args.summary:
        summary(args.raw, args.pre, args.post)
    else:
        process_all(args.raw, args.out, args.pre, args.post)


if __name__ == "__main__":
    main()
