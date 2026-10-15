"""
Construction de séquences de battements consécutifs pour les modèles séquentiels (CNN-LSTM).

Les battements d'un split sont découpés en blocs contigus de T battements (par record). Un bloc est encodé
une seule fois par le CNN ; chaque battement « central » du bloc voit ensuite une fenêtre de K battements
voisins (K = 2h + 1). Pour qu'un central ait toujours sa fenêtre complète dans le bloc, les blocs se
recouvrent de 2h battements, et chaque record est complété par h battements factices (masqués) de chaque côté.
"""
import numpy as np


def build_sequences(split: dict, k: int = 9, t: int = 40) -> dict:
    h, stride = k // 2, t - 2 * (k // 2)
    assert t > 2 * h
    rec, X, y = split["rec"], split["X"], split["y"]
    log_rr = np.log(split["rr"][:, 0]).astype(np.float32)          # RR précédent (s), en log ; centré dans le modèle

    cuts = np.flatnonzero(rec[1:] != rec[:-1]) + 1
    runs = list(zip([0, *cuts], [*cuts, len(rec)]))
    assert len({rec[s] for s, _ in runs}) == len(runs), "un record doit être contigu"

    Xs, lrs, ys, origs, starts, offset = [], [], [], [], [], 0
    for s, e in runs:
        n = e - s
        c = int(np.ceil(n / stride))
        length = 2 * h + c * stride
        xb = np.zeros((length, X.shape[1]), np.float32)
        lb = np.full(length, np.nan, np.float32)
        yb = np.full(length, -1, np.int64)
        ob = np.full(length, -1, np.int64)
        xb[h:h + n], lb[h:h + n], yb[h:h + n], ob[h:h + n] = X[s:e], log_rr[s:e], y[s:e], np.arange(s, e)
        Xs.append(xb), lrs.append(lb), ys.append(yb), origs.append(ob)
        starts.extend(offset + j * stride for j in range(c))
        offset += length
    return {"X": np.concatenate(Xs), "lr": np.concatenate(lrs), "y": np.concatenate(ys),
            "orig": np.concatenate(origs), "starts": np.array(starts, np.int64), "k": k, "t": t, "n": len(y)}
