"""
Augmentation de données pour séries temporelles ECG.

Philosophie du module
---------------------
Les bases de données ECG souffrent généralement d'un fort déséquilibre de classes :
les battements normaux (N) sont 10 à 100 fois plus nombreux que les arythmies rares
mais cliniquement critiques (V, S, F).

Deux stratégies complémentaires sont combinées ici :

  1. **Augmentation géométrique** : transformations qui préservent le sens médical du
     signal (un battement augmenté reste physiquement plausible) :
     - *Jittering* : ajout d'un bruit gaussien faible (simule le bruit de mesure).
     - *Scaling* : multiplication par un scalaire aléatoire (simule la variabilité
       d'amplitude inter-patients).
     - *Time warping* : déformation légère de l'axe temporel (simule les variations
       de fréquence cardiaque pendant l'enregistrement).
     - *Magnitude warping* : application d'une courbe sinusoïdale lisse sur l'amplitude
       (simule les artefacts de respiration).

  2. **Rééquilibrage par sur-échantillonnage** : ``oversample_minority_classes`` génère
     de nouveaux exemples augmentés pour les classes sous-représentées jusqu'à un ratio
     cible configurable par rapport à la classe majoritaire.

Toutes les transformations opèrent de façon *vectorisée* (batch de signaux) pour
rester compatibles avec un DataLoader PyTorch.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Transformations individuelles
# ---------------------------------------------------------------------------

def jitter(X: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Ajoute un bruit gaussien i.i.d. sur chaque échantillon.

    ``sigma`` est choisi empiriquement pour rester sous le seuil de perception
    clinique (< 5 % de l'amplitude typique d'un QRS).
    """
    return X + np.random.normal(0, sigma, X.shape).astype(X.dtype)


def scaling(X: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Multiplie chaque signal par un facteur tiré dans N(1, sigma).

    Préserve la morphologie : un scalaire unique par exemple dans le batch.
    """
    factors = np.random.normal(1.0, sigma, (X.shape[0], 1)).astype(X.dtype)
    return X * factors


def time_warp(X: np.ndarray, sigma: float = 0.2, knots: int = 4) -> np.ndarray:
    """Déforme légèrement l'axe temporel par interpolation cubique.

    On génère une courbe de déformation lisse à partir de ``knots`` points de
    contrôle, puis on ré-échantillonne chaque signal selon cette courbe.  L'effet
    reste subtil (< 20 % de déformation locale) pour conserver la plausibilité
    physiologique.
    """
    from scipy.interpolate import CubicSpline

    T   = X.shape[1]
    t   = np.arange(T, dtype=np.float64)
    out = np.empty_like(X)

    # Points de contrôle uniformément répartis + perturbation aléatoire
    ctrl_t = np.linspace(0, T - 1, knots + 2)
    for i in range(X.shape[0]):
        ctrl_y   = ctrl_t + np.random.normal(0, sigma * T / (knots + 1), ctrl_t.shape)
        ctrl_y   = np.clip(ctrl_y, 0, T - 1)
        ctrl_y[0], ctrl_y[-1] = 0.0, float(T - 1)  # ancrage aux bornes

        warp_fn = CubicSpline(ctrl_t, ctrl_y)
        t_warped = np.clip(warp_fn(t), 0, T - 1)

        # Ré-interpolation du signal sur l'axe déformé
        out[i] = np.interp(t_warped, t, X[i])

    return out.astype(X.dtype)


def magnitude_warp(X: np.ndarray, sigma: float = 0.2, knots: int = 4) -> np.ndarray:
    """Applique une déformation lisse sur l'amplitude.

    Une courbe sinusoïdale composite est générée et multipliée point à point.
    Simule les variations d'amplitude dues à la respiration.
    """
    from scipy.interpolate import CubicSpline

    T    = X.shape[1]
    t    = np.arange(T, dtype=np.float64)
    out  = np.empty_like(X)

    ctrl_t = np.linspace(0, T - 1, knots + 2)
    for i in range(X.shape[0]):
        ctrl_y  = 1.0 + np.random.normal(0, sigma, ctrl_t.shape)
        warp_fn = CubicSpline(ctrl_t, ctrl_y)
        out[i]  = X[i] * warp_fn(t).astype(X.dtype)

    return out


# ---------------------------------------------------------------------------
# Augmentation aléatoire : pipeline combiné
# ---------------------------------------------------------------------------

def random_augment(X: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Applique aléatoirement une ou plusieurs transformations.

    Chaque transformation est appliquée indépendamment avec probabilité ``p``.
    Cette flexibilité permet de créer une grande diversité de patterns tout en
    contrôlant le degré de modification.

    Parameters
    ----------
    X : ndarray (B, T)
        Batch de signaux.
    p : float
        Probabilité d'application de chaque transformation.
    """
    if np.random.rand() < p:
        X = jitter(X)
    if np.random.rand() < p:
        X = scaling(X)
    if np.random.rand() < p:
        X = magnitude_warp(X)
    if np.random.rand() < p:
        X = time_warp(X)
    return X


# ---------------------------------------------------------------------------
# Rééquilibrage par sur-échantillonnage augmenté
# ---------------------------------------------------------------------------

def oversample_minority_classes(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sur-échantillonne les classes minoritaires par augmentation.

    Pour chaque classe minoritaire, on tire aléatoirement des exemples existants
    et on leur applique ``random_augment`` jusqu'à atteindre au moins
    ``target_ratio * n_majority`` exemples.

    Parameters
    ----------
    X : ndarray (N, T)
    y : ndarray (N,)
    target_ratio : float
        Ratio minimum classe_minoritaire / classe_majoritaire après rééquilibrage.
    seed : int

    Returns
    -------
    X_bal, y_bal
        Tableaux rééquilibrés (les exemples originaux sont conservés, les nouveaux
        sont ajoutés à la fin).
    """
    rng = np.random.default_rng(seed)

    classes, counts = np.unique(y, return_counts=True)
    n_majority = counts.max()
    target_n   = int(n_majority * target_ratio)

    X_extra, y_extra = [], []

    for cls, cnt in zip(classes, counts):
        if cnt >= target_n:
            continue  # classe déjà suffisamment représentée

        n_needed  = target_n - cnt
        idx_cls   = np.where(y == cls)[0]

        # Tirage avec remise parmi les exemples de cette classe
        chosen    = rng.choice(idx_cls, size=n_needed, replace=True)
        X_sampled = X[chosen].copy()

        # Augmentation pour éviter les doublons exacts
        X_aug = random_augment(X_sampled, p=0.8)

        X_extra.append(X_aug)
        y_extra.append(np.full(n_needed, cls, dtype=y.dtype))

    if not X_extra:
        return X, y

    X_bal = np.concatenate([X] + X_extra, axis=0)
    y_bal = np.concatenate([y] + y_extra, axis=0)

    # Mélange final pour éviter que tous les exemples synthétiques soient en fin de tableau
    perm  = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]
