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


class CNNLSTMAttention(nn.Module):
    """CNN par battement + BiLSTM sur K battements consécutifs + attention sur la séquence.

    - Encodeur : même tronc convolutif que CNN1D, appliqué à chaque battement (poids partagés).
    - Rythme : pour chaque battement de la fenêtre, log(RR précédent) centré sur la médiane de la fenêtre,
      donc relatif (aucune fréquence cardiaque absolue), projeté puis concaténé à l'embedding de forme.
    - BiLSTM sur les K pas, attention additive masquée (les battements factices en bord de record sont ignorés),
      puis classification à partir de [contexte pondéré ; état du battement central].

    Entrée : x (B, T, L) battements, lr (B, T) log-RR (NaN pour les battements factices), avec T >= K.
    Sortie : logits (B * (T - K + 1), n_classes), un par battement central, dans l'ordre du temps.
    """

    uses_sequence = True

    def __init__(self, n_classes=5, channels=(32, 64, 128), kernels=(7, 5, 3), k=9, rr_dim=16, hidden=64,
                 dropout=0.3):
        super().__init__()
        c1, c2, c3 = channels
        self.k = k
        self.features = nn.Sequential(
            conv_block(1, c1, kernels[0], pool=True),
            conv_block(c1, c2, kernels[1], pool=True),
            conv_block(c2, c3, kernels[2], pool=False),
        )
        self.rr_proj = nn.Sequential(nn.Linear(1, rr_dim), nn.ReLU())
        self.lstm = nn.LSTM(c3 + rr_dim, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_classes))

    def forward(self, x, lr, return_attn=False):
        b, t, length = x.shape
        emb = self.features(x.reshape(b * t, 1, length)).mean(dim=-1).reshape(b, t, -1)
        # fenêtres glissantes de K battements : (n_fenêtres, K, ...)
        ew = emb.unfold(1, self.k, 1).permute(0, 1, 3, 2).reshape(-1, self.k, emb.shape[-1])
        lw = lr.unfold(1, self.k, 1).reshape(-1, self.k)
        mask = ~torch.isnan(lw)
        rel = torch.nan_to_num(lw - torch.nanmedian(lw, dim=1, keepdim=True).values, nan=0.0)
        feats = torch.cat([ew, self.rr_proj(rel.unsqueeze(-1))], dim=-1) * mask.unsqueeze(-1)
        out, _ = self.lstm(feats)
        scores = self.att(out).squeeze(-1).masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=1)
        ctx = (alpha.unsqueeze(-1) * out).sum(dim=1)
        logits = self.head(torch.cat([ctx, out[:, self.k // 2]], dim=1))
        return (logits, alpha) if return_attn else logits
