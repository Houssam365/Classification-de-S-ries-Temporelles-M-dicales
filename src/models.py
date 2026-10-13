"""
Architectures de réseaux de neurones pour la classification ECG.

Philosophie du module
---------------------
Trois familles d'architectures sont implémentées, du plus simple au plus complexe,
pour permettre une comparaison rigoureuse (ablation study) :

  1. **CNN1D** – Réseau convolutif pur : idéal pour capturer des motifs locaux
     (forme du QRS, durée du segment ST). Architecture de référence légère.

  2. **ResNet1D** – ResNet adapté aux séries temporelles 1D (inspiré de Wang et al.
     2017).  Les connexions résiduelles (skip connections) permettent d'entraîner
     des réseaux profonds (> 10 couches) sans dégradation du gradient.

  3. **CNN_LSTM** – Architecture hybride : le CNN extrait des caractéristiques
     morphologiques locales, le LSTM capture les dépendances temporelles à long terme
     (relations entre battements successifs).  Un mécanisme d'**attention** pondère
     les pas de temps pertinents avant la classification.

Chaque architecture hérite de ``nn.Module`` et expose une signature commune :
  - entrée : ``(batch, 1, T)`` – signal ECG mono-canal
  - sortie : ``(batch, n_classes)`` – logits non normalisés

Baseline Machine Learning
--------------------------
``HandcraftedFeatureExtractor`` extrait manuellement des statistiques classiques
(intervalles RR, morphologie QRS, énergie…) pour entraîner un Random Forest.
Cette baseline permet de quantifier l'apport des approches deep learning.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture 1 : CNN1D – référence légère
# ---------------------------------------------------------------------------

class CNN1D(nn.Module):
    """Réseau convolutif 1D à 3 blocs convolutifs suivi d'un classifieur.

    Structure :
      [Conv → BN → ReLU → MaxPool] × 3  →  Global Average Pooling  →  FC

    Le Global Average Pooling (GAP) remplace les couches fully-connected sur les
    features maps, réduisant drastiquement le nombre de paramètres et améliorant
    la généralisation.
    """

    def __init__(self, n_classes: int = 5, window_size: int = 360) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloc 1 : filtres larges pour capturer l'onde P et le QRS
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T/2 = 180

            # Bloc 2 : filtres moyens pour affiner les détails morphologiques
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T/4 = 90

            # Bloc 3 : filtres fins pour les structures à haute fréquence
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T/8 = 45
        )

        # Global Average Pooling : moyenne sur la dimension temporelle
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 1, T)
        x = self.features(x)   # (B, 128, T')
        x = self.gap(x)        # (B, 128, 1)
        return self.classifier(x)  # (B, n_classes)


# ---------------------------------------------------------------------------
# Architecture 2 : ResNet1D – réseau profond avec connexions résiduelles
# ---------------------------------------------------------------------------

class _ResBlock1D(nn.Module):
    """Bloc résiduel élémentaire pour séries temporelles 1D.

    Deux couches convolutives avec BN et ReLU, plus une connexion de raccourci
    (shortcut / skip connection).  Si les dimensions d'entrée et de sortie
    diffèrent, le shortcut est adapté par une convolution 1×1.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )

        # Projection pour aligner les dimensions du shortcut si nécessaire
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv_block(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    """ResNet 1D adapté à la classification de séries temporelles ECG.

    Architecture inspirée de Wang et al. (2017) : 3 blocs résiduels avec
    un nombre croissant de canaux (64→128→256), suivi d'un GAP et d'un classifieur.

    Les connexions résiduelles résolvent le problème de dégradation du gradient,
    permettant d'empiler des dizaines de couches sans perte de performance.
    """

    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()

        # Couche d'entrée : projection vers l'espace de features initial
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Blocs résiduels avec augmentation progressive du nombre de filtres
        self.layer1 = nn.Sequential(_ResBlock1D(64, 64), _ResBlock1D(64, 64))
        self.layer2 = nn.Sequential(_ResBlock1D(64, 128), _ResBlock1D(128, 128))
        self.layer3 = nn.Sequential(_ResBlock1D(128, 256), _ResBlock1D(256, 256))

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Architecture 3 : CNN-LSTM avec attention
# ---------------------------------------------------------------------------

class _BahdanauAttention(nn.Module):
    """Mécanisme d'attention de Bahdanau (additive attention).

    Pour chaque pas de temps du LSTM, un score d'attention est calculé.
    La représentation finale est la somme pondérée des états cachés.

    L'attention permet au modèle d'identifier *quels instants du signal*
    sont les plus discriminants pour la décision de classification
    (utile pour l'interprétabilité clinique).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h : (B, T', hidden_size)
        scores = self.v(torch.tanh(self.w(h))).squeeze(-1)  # (B, T')
        weights = F.softmax(scores, dim=-1)                  # (B, T')
        context = (weights.unsqueeze(-1) * h).sum(dim=1)    # (B, hidden_size)
        return context, weights


class CNN_LSTM(nn.Module):
    """Architecture hybride : extraction CNN + modélisation temporelle LSTM + attention.

    Étapes :
      1. **CNN** : extrait des représentations locales de chaque segment du signal
         (équivalent à des « patches » morphologiques).
      2. **LSTM bidirectionnel** : modélise les dépendances temporelles entre patches
         dans les deux sens (passé→futur et futur→passé).
      3. **Attention** : pondère les pas de temps selon leur pertinence.
      4. **Classifieur** : produit les logits finaux.

    Le LSTM bidirectionnel est particulièrement adapté aux signaux ECG car certaines
    anomalies (ex. flutter atrial) se caractérisent par des motifs *récurrents*
    à travers le signal, que le LSTM peut capturer efficacement.
    """

    def __init__(
        self,
        n_classes: int = 5,
        cnn_out_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Extraction de features locales par CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T/2

            nn.Conv1d(32, cnn_out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T/4
        )

        # LSTM bidirectionnel sur la séquence de features CNN
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Attention sur les sorties du LSTM
        self.attention = _BahdanauAttention(lstm_hidden * 2)  # ×2 car bidirectionnel

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns
        -------
        logits : (B, n_classes)
        attn_weights : (B, T') — poids d'attention pour visualisation (interprétabilité)
        """
        # Extraction CNN : (B, 1, T) → (B, C, T/4)
        feat = self.cnn(x)

        # Transposition pour le LSTM : (B, C, T') → (B, T', C)
        feat = feat.permute(0, 2, 1)

        # LSTM bidirectionnel : (B, T', C) → (B, T', 2H)
        h_seq, _ = self.lstm(feat)

        # Attention + agrégation : (B, T', 2H) → (B, 2H)
        context, attn_weights = self.attention(h_seq)

        logits = self.classifier(context)
        return logits, attn_weights


# ---------------------------------------------------------------------------
# Baseline Machine Learning : extraction de features manuelles
# ---------------------------------------------------------------------------

class HandcraftedFeatureExtractor:
    """Extrait des features statistiques et morphologiques pour un classifieur ML.

    Ces features reproduisent l'expertise cardiologique traditionnelle :
    - Statistiques d'ordre 1–4 (moyenne, variance, skewness, kurtosis)
    - Énergie et puissance du signal
    - Statistiques sur les intervalles entre pics (RR variability)
    - Nombre de passages par zéro (zero-crossing rate)
    - Énergie dans des bandes fréquentielles (via FFT)

    Combiné à un Random Forest, ceci constitue une baseline solide et
    interprétable avant d'introduire les modèles deep learning.
    """

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : ndarray (N, T)

        Returns
        -------
        features : ndarray (N, D)  avec D = nombre de features extraites
        """
        feats = []
        for signal in X:
            f = self._extract_one(signal)
            feats.append(f)
        return np.stack(feats, axis=0)

    def _extract_one(self, s: np.ndarray) -> np.ndarray:
        """Extrait les features d'un seul signal de longueur T."""
        features: List[float] = []

        # --- Statistiques temporelles ---
        features += [
            float(np.mean(s)),
            float(np.std(s)),
            float(np.max(s)),
            float(np.min(s)),
            float(np.max(s) - np.min(s)),       # amplitude crête-à-crête
        ]

        # Skewness et Kurtosis (asymétrie et aplatissement de la distribution)
        std = float(np.std(s)) + 1e-8
        mu  = float(np.mean(s))
        features += [
            float(np.mean(((s - mu) / std) ** 3)),  # skewness
            float(np.mean(((s - mu) / std) ** 4)),  # kurtosis
        ]

        # --- Énergie ---
        features.append(float(np.mean(s ** 2)))    # énergie moyenne

        # --- Zero-crossing rate ---
        zcr = np.sum(np.diff(np.sign(s)) != 0) / len(s)
        features.append(float(zcr))

        # --- Domaine fréquentiel (FFT) ---
        fft_mag = np.abs(np.fft.rfft(s))
        T = len(fft_mag)
        # Énergie dans 4 bandes fréquentielles équi-espacées (0–90 Hz à 360 Hz)
        for i in range(4):
            lo, hi = int(i * T / 4), int((i + 1) * T / 4)
            features.append(float(np.sum(fft_mag[lo:hi] ** 2)))
        # Fréquence dominante
        features.append(float(np.argmax(fft_mag)))

        return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Utilitaire : instanciation par nom
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "cnn1d": CNN1D,
    "resnet1d": ResNet1D,
    "cnn_lstm": CNN_LSTM,
}


def build_model(name: str, n_classes: int = 5, **kwargs) -> nn.Module:
    """Instancie un modèle par son nom (clé dans MODEL_REGISTRY)."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modèle '{name}' inconnu. Choisir parmi : {list(MODEL_REGISTRY)}."
        )
    return MODEL_REGISTRY[name](n_classes=n_classes, **kwargs)
