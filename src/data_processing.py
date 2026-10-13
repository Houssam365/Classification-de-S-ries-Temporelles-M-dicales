"""
Chargement, segmentation et normalisation des signaux ECG MIT-BIH.

Philosophie du module
---------------------
La base MIT-BIH contient des enregistrements ECG continus annotés battement par
battement. Pour entraîner un classifieur, il faut transformer ces signaux bruts en
un jeu d'exemples (X, y) : chaque exemple est une fenêtre de signal centrée sur un
battement, et son étiquette est le type d'arythmie annoté.

Ce module expose deux objets principaux :

  - ``AAMI_CLASSES`` : mapping des symboles WFDB → 5 classes standardisées AAMI
    (norme internationale reconnue par les cardiologues).
  - ``ECGDataset`` : classe qui orchestre le chargement des enregistrements décodés,
    la segmentation par battement et la normalisation patient-par-patient.

Choix techniques importants
----------------------------
- **Fenêtre de 360 échantillons (1 s à 360 Hz)** : couvre un cycle cardiaque complet
  avec une marge confortable avant/après chaque battement.
- **Stratification par patient** : on évite la fuite de données (data leakage) en
  s'assurant que les enregistrements d'un même patient ne se retrouvent pas à la fois
  dans train et test.
- **Normalisation par enregistrement** : chaque enregistrement est normalisé
  indépendamment (z-score sur le signal entier) pour atténuer les variations
  inter-patients dues au matériel ou à la position des électrodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Correspondance symboles WFDB → classes AAMI
# ---------------------------------------------------------------------------
# La norme AAMI EC57 regroupe les 40+ symboles d'annotation MIT-BIH en 5
# super-classes pertinentes cliniquement.  Les battements non classifiables
# (pacemaker, bruit…) sont ignorés.
AAMI_CLASSES: Dict[str, int] = {
    # N – Normal and bundle-branch block beats
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    # S – Supraventricular ectopic beats
    "A": 1, "a": 1, "J": 1, "S": 1,
    # V – Ventricular ectopic beats
    "V": 2, "E": 2,
    # F – Fusion beats
    "F": 3,
    # Q – Unknown / paced beats
    "Q": 4, "/": 4,
}

CLASS_NAMES = ["N (Normal)", "S (Supra-V)", "V (Ventric.)", "F (Fusion)", "Q (Paced)"]

# Fenêtre : 180 échantillons avant le pic R, 180 après → 360 total (1 seconde)
WINDOW_BEFORE = 180
WINDOW_AFTER  = 180
WINDOW_SIZE   = WINDOW_BEFORE + WINDOW_AFTER  # 360 samples


# ---------------------------------------------------------------------------
# Chargement d'un seul enregistrement NPZ
# ---------------------------------------------------------------------------

def _load_record(npz_path: Path, lead_index: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Charge un enregistrement décodé et retourne (signal_1D, samples, symboles).

    Parameters
    ----------
    npz_path:
        Chemin vers le fichier ``.npz`` produit par ``scripts/download_data.py``.
    lead_index:
        Index de la dérivation à utiliser (0 = MLII par défaut dans MIT-BIH).

    Returns
    -------
    signal : ndarray (T,)
        Signal ECG mono-canal normalisé en z-score.
    ann_samples : ndarray (B,)
        Positions (en samples) des pics R annotés.
    ann_symbols : ndarray (B,)
        Symboles d'annotation associés (e.g. 'N', 'V', 'A'…).
    """
    with np.load(npz_path, allow_pickle=False) as f:
        signal = f["signal"][:, lead_index].astype(np.float32)  # (T,)
        ann_samples = f["annotation_samples"].astype(np.int64)
        ann_symbols = f["annotation_symbols"].astype(str)

    # Normalisation z-score sur l'enregistrement entier
    mu, sigma = signal.mean(), signal.std()
    if sigma > 1e-8:
        signal = (signal - mu) / sigma

    return signal, ann_samples, ann_symbols


# ---------------------------------------------------------------------------
# Segmentation : un battement → une fenêtre de taille fixe
# ---------------------------------------------------------------------------

def _segment_beats(
    signal: np.ndarray,
    ann_samples: np.ndarray,
    ann_symbols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrait les fenêtres centrées sur les pics R et leurs étiquettes AAMI.

    Les battements trop proches du début ou de la fin du signal, ainsi que les
    symboles hors de la taxonomie AAMI, sont silencieusement ignorés.

    Returns
    -------
    beats : ndarray (B, WINDOW_SIZE)
    labels : ndarray (B,)  — entiers dans {0, 1, 2, 3, 4}
    """
    beats, labels = [], []
    T = len(signal)

    for sample, symbol in zip(ann_samples, ann_symbols):
        # Ignorer les symboles non reconnus par la norme AAMI
        if symbol not in AAMI_CLASSES:
            continue

        start = sample - WINDOW_BEFORE
        end   = sample + WINDOW_AFTER

        # Ignorer les fenêtres qui débordent du signal
        if start < 0 or end > T:
            continue

        beats.append(signal[start:end])
        labels.append(AAMI_CLASSES[symbol])

    if not beats:
        return np.empty((0, WINDOW_SIZE), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(beats, axis=0), np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Classe principale : ECGDataset
# ---------------------------------------------------------------------------

class ECGDataset:
    """Gère le chargement, la segmentation et le découpage train/val/test.

    Paramètres
    ----------
    decoded_dir : str | Path
        Répertoire contenant les fichiers ``*.npz`` produits par download_data.py.
    val_ratio : float
        Fraction des enregistrements réservés à la validation.
    test_ratio : float
        Fraction des enregistrements réservés au test.
    seed : int
        Graine aléatoire pour la reproductibilité du découpage.
    lead_index : int
        Index de la dérivation ECG à utiliser.
    """

    def __init__(
        self,
        decoded_dir: str | Path,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        lead_index: int = 0,
    ) -> None:
        self.decoded_dir = Path(decoded_dir).expanduser().resolve()
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio
        self.seed        = seed
        self.lead_index  = lead_index

        self._npz_paths: List[Path] = sorted(self.decoded_dir.glob("*.npz"))
        if not self._npz_paths:
            raise FileNotFoundError(
                f"Aucun fichier .npz trouvé dans {self.decoded_dir}. "
                "Lancez d'abord `python scripts/download_data.py`."
            )

        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._record_ids: Optional[np.ndarray] = None  # id d'enregistrement pour chaque beat

    # ------------------------------------------------------------------
    # Chargement et segmentation de tous les enregistrements
    # ------------------------------------------------------------------

    def load(self) -> "ECGDataset":
        """Charge et segmente tous les enregistrements. Retourne self (chaînable)."""
        all_X, all_y, all_ids = [], [], []

        for idx, npz_path in enumerate(self._npz_paths):
            signal, ann_samples, ann_symbols = _load_record(npz_path, self.lead_index)
            X_rec, y_rec = _segment_beats(signal, ann_samples, ann_symbols)

            if len(X_rec) == 0:
                continue

            all_X.append(X_rec)
            all_y.append(y_rec)
            all_ids.append(np.full(len(X_rec), idx, dtype=np.int32))

        self._X          = np.concatenate(all_X, axis=0)   # (N, 360)
        self._y          = np.concatenate(all_y, axis=0)   # (N,)
        self._record_ids = np.concatenate(all_ids, axis=0) # (N,)
        return self

    # ------------------------------------------------------------------
    # Découpage train / val / test stratifié PAR PATIENT
    # ------------------------------------------------------------------

    def split(self) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """Découpe en train / val / test en respectant les frontières patient.

        IMPORTANT : la stratification est réalisée au niveau de l'enregistrement
        (i.e. du patient), pas au niveau du battement.  Cela évite que le modèle
        mémorise les caractéristiques individuelles d'un patient présent à la fois
        dans train et test (data leakage).

        Returns
        -------
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        if self._X is None:
            raise RuntimeError("Appelez d'abord `dataset.load()`.")

        rng = np.random.default_rng(self.seed)

        unique_ids = np.unique(self._record_ids)
        n_records  = len(unique_ids)

        # Mélange aléatoire des enregistrements
        shuffled = rng.permutation(unique_ids)

        n_test = max(1, int(n_records * self.test_ratio))
        n_val  = max(1, int(n_records * self.val_ratio))

        test_ids  = set(shuffled[:n_test])
        val_ids   = set(shuffled[n_test: n_test + n_val])
        train_ids = set(shuffled[n_test + n_val:])

        def _mask(id_set):
            return np.isin(self._record_ids, list(id_set))

        train_mask = _mask(train_ids)
        val_mask   = _mask(val_ids)
        test_mask  = _mask(test_ids)

        return (
            (self._X[train_mask], self._y[train_mask]),
            (self._X[val_mask],   self._y[val_mask]),
            (self._X[test_mask],  self._y[test_mask]),
        )

    # ------------------------------------------------------------------
    # Statistiques descriptives
    # ------------------------------------------------------------------

    def describe(self) -> pd.DataFrame:
        """Retourne un tableau descriptif du déséquilibre de classes."""
        if self._y is None:
            raise RuntimeError("Appelez d'abord `dataset.load()`.")

        total = len(self._y)
        rows = []
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            count = int((self._y == cls_idx).sum())
            rows.append({
                "Classe": cls_name,
                "Indice": cls_idx,
                "Nombre de beats": count,
                "Proportion (%)": round(100 * count / total, 2),
            })
        return pd.DataFrame(rows)
