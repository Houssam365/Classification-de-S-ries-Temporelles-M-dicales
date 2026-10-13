"""
Interprétabilité des modèles de classification ECG.

Philosophie du module
---------------------
Dans un contexte médical, un modèle qui prédit bien mais dont les décisions
sont opaques est difficile à valider cliniquement.  Ce module implémente trois
approches complémentaires d'interprétabilité :

  1. **GradCAM temporel (Gradient-weighted Class Activation Mapping)** :
     Technique issue de la vision par ordinateur, adaptée aux séries temporelles 1D.
     Elle répond à la question : « Quelles parties du signal ECG ont le plus influencé
     la décision du modèle ? »  Pratique : on calcule le gradient de la classe prédite
     par rapport aux activations de la dernière couche convolutive ; les zones avec un
     gradient élevé sont les plus « importantes » pour la décision.

  2. **Visualisation des poids d'attention (CNN-LSTM)** :
     Le mécanisme d'attention du CNN-LSTM apprend explicitement à pondérer les pas
     de temps.  On peut afficher cette carte d'attention directement sur le signal
     pour identifier les instants critiques (ex. pic R, segment ST).

  3. **Visualisation des filtres convolutifs** :
     Les premiers filtres d'un CNN 1D apprennent des détecteurs de motifs élémentaires
     (analogues aux détecteurs de bords en vision).  Leur visualisation révèle quelles
     formes le réseau cherche dans le signal ECG.

Ces outils sont indispensables pour la **validation clinique** : un cardiologue peut
vérifier que les régions mises en avant correspondent à des caractéristiques connues
(pic R, onde T, segment ST) et non à des artefacts.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. GradCAM 1D
# ---------------------------------------------------------------------------

class GradCAM1D:
    """Calcule la carte d'activation GradCAM pour une couche convolutive cible.

    Paramètres
    ----------
    model : nn.Module
        Modèle entraîné (CNN1D ou ResNet1D).
    target_layer : nn.Module
        Couche convolutive à « inspecter ».  Typiquement la dernière Conv1d du réseau.
    device : torch.device

    Utilisation
    -----------
    >>> gradcam = GradCAM1D(model, target_layer=model.features[-1])
    >>> heatmap = gradcam.compute(x)  # x : (1, 1, T)
    """

    def __init__(
        self,
        model:        nn.Module,
        target_layer: nn.Module,
        device:       Optional[torch.device] = None,
    ) -> None:
        self.model        = model
        self.target_layer = target_layer
        self.device       = device or torch.device("cpu")

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor]   = None

        # Hooks pour capturer les activations forward et les gradients backward
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def compute(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Calcule la heatmap GradCAM.

        Parameters
        ----------
        x : tensor (1, 1, T) — un seul exemple
        class_idx : int optionnel — si None, utilise la classe prédite.

        Returns
        -------
        heatmap : ndarray (T,) — activations GradCAM interpolées sur la longueur du signal
        """
        self.model.eval()
        x = x.to(self.device).requires_grad_(True)

        out = self.model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Rétropropagation ciblée sur la classe d'intérêt
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=False)

        # Pondération des activations par les gradients moyennés spatialement
        # (Global Average Pooling sur les gradients)
        weights = self._gradients.mean(dim=-1, keepdim=True)  # (1, C, 1)
        cam     = (weights * self._activations).sum(dim=1)     # (1, T')
        cam     = F.relu(cam).squeeze(0).cpu().numpy()         # (T',)

        # Interpolation sur la longueur originale du signal
        T_orig  = x.shape[-1]
        heatmap = np.interp(
            np.linspace(0, len(cam) - 1, T_orig),
            np.arange(len(cam)),
            cam,
        )

        # Normalisation [0, 1]
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)

        return heatmap

    def remove_hooks(self) -> None:
        """Libère les hooks (important pour éviter les fuites mémoire)."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def plot_gradcam(
    signal:    np.ndarray,
    heatmap:   np.ndarray,
    fs:        float = 360.0,
    title:     str = "GradCAM — Régions discriminantes",
    save_path: Optional[str] = None,
) -> None:
    """Superpose la heatmap GradCAM sur le signal ECG.

    Le signal est affiché en noir, la heatmap en couleur (rouge = zone critique
    pour la décision du modèle).
    """
    t = np.arange(len(signal)) / fs  # axe temporel en secondes

    fig, ax = plt.subplots(figsize=(12, 4))

    # Signal ECG
    ax.plot(t, signal, color="black", lw=1.2, zorder=2, label="Signal ECG")

    # Heatmap : coloration de fond par intensité GradCAM
    colormap = cm.get_cmap("RdYlGn_r")
    for i in range(len(t) - 1):
        ax.axvspan(t[i], t[i + 1], alpha=0.4 * heatmap[i], color=colormap(heatmap[i]))

    ax.set_xlabel("Temps (s)", fontsize=11)
    ax.set_ylabel("Amplitude (normalisée)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Importance GradCAM")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Visualisation de l'attention (CNN-LSTM)
# ---------------------------------------------------------------------------

def plot_attention_weights(
    signal:       np.ndarray,
    attn_weights: np.ndarray,
    fs:           float = 360.0,
    title:        str = "Poids d'attention — CNN-LSTM",
    save_path:    Optional[str] = None,
) -> None:
    """Affiche les poids d'attention du modèle CNN-LSTM sur le signal ECG.

    Les poids d'attention sont interpolés sur la longueur du signal original
    (ils sont calculés sur les features CNN, donc à une résolution réduite).

    Parameters
    ----------
    signal : ndarray (T,)
    attn_weights : ndarray (T',) — poids retournés par CNN_LSTM.forward()
    fs : float — fréquence d'échantillonnage en Hz
    """
    T = len(signal)
    t = np.arange(T) / fs

    # Interpolation des poids d'attention à la résolution du signal
    attn_interp = np.interp(
        np.linspace(0, len(attn_weights) - 1, T),
        np.arange(len(attn_weights)),
        attn_weights,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(t, signal, color="steelblue", lw=1.2)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3)

    ax2.fill_between(t, attn_interp, alpha=0.7, color="tomato")
    ax2.plot(t, attn_interp, color="darkred", lw=0.8)
    ax2.set_ylabel("Poids d'attention")
    ax2.set_xlabel("Temps (s)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Visualisation des filtres convolutifs
# ---------------------------------------------------------------------------

def plot_conv_filters(
    model:     nn.Module,
    layer_idx: int = 0,
    n_filters: int = 16,
    save_path: Optional[str] = None,
) -> None:
    """Visualise les filtres de la première couche convolutive d'un CNN1D.

    Les filtres de la première couche sont directement dans l'espace du signal ECG :
    on peut les interpréter comme des « motifs cherchés » (détecteurs de pics, de
    plateaux, de transition…).

    Parameters
    ----------
    model : CNN1D ou ResNet1D
    layer_idx : int — index de la Conv1d dans le Sequential (souvent 0)
    n_filters : int — nombre de filtres à afficher
    """
    # Extraction des poids de la première couche convolutive
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            first_conv = module
            break

    if first_conv is None:
        raise ValueError("Aucune couche Conv1d trouvée dans le modèle.")

    weights = first_conv.weight.data.cpu().numpy()  # (out_ch, in_ch, kernel_size)
    n_show  = min(n_filters, weights.shape[0])

    fig, axes = plt.subplots(4, n_show // 4, figsize=(14, 6))
    axes = axes.flatten()

    for i in range(n_show):
        filt = weights[i, 0]  # filtre i, canal 0
        axes[i].plot(filt, color="navy", lw=1.2)
        axes[i].set_title(f"F{i}", fontsize=8)
        axes[i].axis("off")

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Filtres convolutifs — 1ère couche CNN", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Analyse des exemples mal classés
# ---------------------------------------------------------------------------

def plot_misclassified(
    X_test:      np.ndarray,
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list,
    fs:          float = 360.0,
    n_examples:  int = 6,
    save_path:   Optional[str] = None,
) -> None:
    """Affiche des exemples mal classés pour analyse d'erreurs.

    L'analyse des erreurs les plus fréquentes est essentielle pour comprendre
    les limites du modèle et guider les améliorations futures.
    """
    wrong_mask  = y_true != y_pred
    wrong_idx   = np.where(wrong_mask)[0]

    if len(wrong_idx) == 0:
        print("Aucune erreur de classification détectée sur ce jeu de données.")
        return

    n_show = min(n_examples, len(wrong_idx))
    chosen = wrong_idx[:n_show]

    fig, axes = plt.subplots(2, n_show // 2, figsize=(14, 6))
    axes = axes.flatten()
    t    = np.arange(X_test.shape[1]) / fs

    for k, idx in enumerate(chosen):
        ax = axes[k]
        ax.plot(t, X_test[idx], color="steelblue", lw=1.0)
        true_name = class_names[y_true[idx]]
        pred_name = class_names[y_pred[idx]]
        ax.set_title(f"Vrai: {true_name}\nPrédit: {pred_name}", fontsize=9, color="darkred")
        ax.grid(alpha=0.3)

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Exemples mal classés", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
