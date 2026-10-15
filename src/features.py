"""
Features hand-crafted pour la baseline Random Forest : morphologie du battement (QRS, P, T,
spectre) et rythme (intervalles RR). Tout est vectorisé sur un tableau (N, longueur de fenêtre).

Le pic R est supposé à l'indice `pre` de chaque fenêtre (cf. data_processing.segment_record).
"""
import numpy as np
from scipy.stats import kurtosis, skew

from src.data_processing import FS

QRS_HALF = 50     # samples (~139 ms) autour du R pour mesurer la largeur du QRS
MORPH_NAMES = [
    "r_amp", "r_abs", "q_min", "s_min", "slope_up", "slope_down", "qrs_width_ms", "qrs_fwhm_ms",
    "qrs_area", "qrs_energy_ratio", "p_ptp", "p_std", "t_max", "t_min", "t_signed", "t_std",
    "post_qrs_mean", "skew", "kurtosis", "band_0.5_5", "band_5_15", "band_15_40",
]
RR_NAMES = ["rr_pre", "rr_post", "rr_local", "rr_pre_over_local", "rr_post_over_local", "rr_pre_over_post"]


def morphology_features(X: np.ndarray, pre: int, fs: int = FS) -> np.ndarray:
    n, w = X.shape
    rows = np.arange(n)
    x = X - np.median(X, axis=1, keepdims=True)

    # Pic R (extremum en valeur absolue autour de la position annotée) ; on retourne le signal
    # pour que la déflexion principale soit positive, y compris pour les battements inversés.
    c = x[:, pre - 5:pre + 6]
    r_pos = pre - 5 + np.abs(c).argmax(axis=1)
    r_amp = x[rows, r_pos]
    y = x * np.where(r_amp < 0, -1.0, 1.0)[:, None]
    r_abs = np.abs(r_amp)

    def seg(a, b):
        return y[:, max(a, 0):min(b, w)]

    q_min = seg(pre - 30, pre).min(axis=1)
    s_min = seg(pre, pre + 40).min(axis=1)

    d = np.diff(y, axis=1) * fs
    slope_up = d[:, pre - 30:pre + 5].max(axis=1)
    slope_down = d[:, pre:pre + 40].min(axis=1)

    # Largeur du QRS : étendue des samples où la pente dépasse 15 % de la pente maximale
    dq = np.abs(d[:, pre - QRS_HALF:pre + QRS_HALF])
    mask = dq > 0.15 * dq.max(axis=1, keepdims=True)
    first = mask.argmax(axis=1)
    last = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    qrs_width = (last - first) / fs * 1000

    # Largeur à mi-hauteur de la déflexion principale
    core = seg(pre - QRS_HALF, pre + QRS_HALF)
    qrs_fwhm = (core > 0.5 * r_abs[:, None]).sum(axis=1) / fs * 1000

    qrs = seg(pre - 25, pre + 25)
    qrs_area = np.abs(qrs).sum(axis=1) / fs
    qrs_energy_ratio = (qrs ** 2).sum(axis=1) / ((y ** 2).sum(axis=1) + 1e-9)

    p = seg(0, pre - 30)                      # région de l'onde P
    t = seg(pre + 40, w)                      # région de l'onde T
    t_max, t_min = t.max(axis=1), t.min(axis=1)
    t_signed = np.where(t_max >= -t_min, t_max, t_min)   # extremum de plus grande amplitude, signé
    post_qrs_mean = seg(pre + 10, pre + 45).mean(axis=1)

    spec = np.abs(np.fft.rfft(y * np.hanning(w), axis=1)) ** 2
    f = np.fft.rfftfreq(w, 1 / fs)
    band = lambda lo, hi: spec[:, (f >= lo) & (f < hi)].sum(axis=1)
    total = band(0.5, 40) + 1e-9
    bands = [band(0.5, 5) / total, band(5, 15) / total, band(15, 40) / total]

    feats = [r_amp, r_abs, q_min, s_min, slope_up, slope_down, qrs_width, qrs_fwhm, qrs_area,
             qrs_energy_ratio, np.ptp(p, axis=1), p.std(axis=1), t_max, t_min, t_signed, t.std(axis=1),
             post_qrs_mean, skew(y, axis=1), kurtosis(y, axis=1), *bands]
    out = np.stack(feats, axis=1).astype(np.float32)
    assert out.shape[1] == len(MORPH_NAMES)
    return out


def rr_feature_matrix(rr: np.ndarray) -> np.ndarray:
    """rr : (N, 4) = [pre, post, local, pre/local] (cf. data_processing.rr_features)."""
    pre, post, local = rr[:, 0], rr[:, 1], rr[:, 2]
    out = np.stack([pre, post, local, rr[:, 3], post / local, pre / post], axis=1).astype(np.float32)
    assert out.shape[1] == len(RR_NAMES)
    return out


def build_features(split: dict, pre: int, kind: str = "all"):
    """kind : 'morph' | 'rr' | 'all'. Retourne (matrice, noms)."""
    parts, names = [], []
    if kind in ("morph", "all"):
        parts.append(morphology_features(split["X"], pre))
        names += MORPH_NAMES
    if kind in ("rr", "all"):
        parts.append(rr_feature_matrix(split["rr"]))
        names += RR_NAMES
    X = np.concatenate(parts, axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names
