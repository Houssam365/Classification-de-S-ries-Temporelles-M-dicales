"""
Architectures : baseline 1D-CNN (inspirée du FCN de Wang et al., 2017, version réduite).
"""
import torch
from torch import nn


def conv_block(c_in, c_out, k, pool):
    layers = [nn.Conv1d(c_in, c_out, k, padding=k // 2, bias=False), nn.BatchNorm1d(c_out), nn.ReLU()]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class CNN1D(nn.Module):
    """3 blocs Conv-BN-ReLU, global average pooling, couche dense.

    Entrée : (batch, 1, longueur) ; sortie : logits (batch, n_classes).
    Le global average pooling garde une carte d'activation temporelle (`features`) exploitable par Grad-CAM.
    """

    def __init__(self, n_classes=5, channels=(32, 64, 128), kernels=(7, 5, 3), dropout=0.3):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            conv_block(1, c1, kernels[0], pool=True),
            conv_block(c1, c2, kernels[1], pool=True),
            conv_block(c2, c3, kernels[2], pool=False),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c3, n_classes))

    uses_rhythm = False

    def forward(self, x):
        return self.head(self.features(x).mean(dim=-1))


class CNNRhythm(nn.Module):
    """CNN1D + branche rythme : fusion tardive entre l'embedding de forme et les RR relatifs.

    Entrées : x (batch, 1, longueur) et r (batch, n_rhythm) = log-ratios RR standardisés
    (précédent/moyenne locale, suivant/moyenne locale, précédent/suivant). Le tronc convolutif est
    identique à CNN1D : seule la branche rythme diffère, ce qui isole son effet.
    """

    uses_rhythm = True

    def __init__(self, n_classes=5, channels=(32, 64, 128), kernels=(7, 5, 3), dropout=0.3,
                 n_rhythm=3, rhythm_dim=16):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            conv_block(1, c1, kernels[0], pool=True),
            conv_block(c1, c2, kernels[1], pool=True),
            conv_block(c2, c3, kernels[2], pool=False),
        )
        self.rhythm = nn.Sequential(nn.Linear(n_rhythm, rhythm_dim), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c3 + rhythm_dim, n_classes))

    def forward(self, x, r):
        z = torch.cat([self.features(x).mean(dim=-1), self.rhythm(r)], dim=1)
        return self.head(z)
