"""
Évaluation des modèles de classification ECG : métriques médicales et cliniques.

Philosophie du module
---------------------
En contexte médical, l'exactitude globale (accuracy) est une métrique trompeuse
sur des données déséquilibrées : un modèle naïf qui prédit toujours "Normal" peut
atteindre 90 % d'accuracy tout en ratant 100 % des arythmies critiques.

Ce module calcule un ensemble de métriques cliniquement pertinentes :

  - **Sensibilité (Recall / Se)** : fraction des vrais arythmies détectées.
    → Priorité absolue : ne pas manquer une arythmie grave.

  - **Spécificité (Sp)** : fraction des battements normaux correctement identifiés.
    → Éviter les fausses alarmes qui épuisent les équipes soignantes.

  - **Valeur Prédictive Positive (VPP / Précision)** : quand le modèle dit
    "arythmie", quelle est la probabilité qu'il ait raison ?

  - **F1-score** : moyenne harmonique précision/rappel, équilibre entre Se et VPP.

  - **Macro F1** : F1 moyen sur toutes les classes, pondération uniforme →
    donne autant d'importance aux classes rares qu'aux classes majoritaires.

  - **Matrice de confusion** : permet d'identifier les confusions inter-classes
    les plus fréquentes (ex. V confondu avec N = erreur grave).

Toutes ces métriques sont calculées par classe et globalement, conformément
à la norme AAMI EC57 pour les évaluateurs de détecteurs d'arythmies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Inférence sur le jeu de test
# ---------------------------------------------------------------------------

def predict(
    model: nn.Module,
    X_test: np.ndarray,
    device: str | torch.device = "auto",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Produit les prédictions et probabilités du modèle sur X_test.

    Parameters
    ----------
    model : nn.Module (déjà chargé avec les poids du meilleur checkpoint)
    X_test : ndarray (N, T)
    device : 'auto', 'cpu', ou 'cuda'
    batch_size : int

    Returns
    -------
    y_pred : ndarray (N,) — classe prédite
    y_prob : ndarray (N, C) — probabilités Softmax par classe
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device).eval()

    X_t  = torch.from_numpy(X_test).unsqueeze(1).float()
    ds   = TensorDataset(X_t)
    dl   = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for (x_batch,) in dl:
            x_batch = x_batch.to(device)
            out     = model(x_batch)
            logits  = out[0] if isinstance(out, (tuple, list)) else out
            probs   = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)  # (N, C)
    y_pred = y_prob.argmax(axis=1)               # (N,)
    return y_pred, y_prob


# ---------------------------------------------------------------------------
# Calcul des métriques médicales
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    class_names: Optional[list] = None,
    n_classes: int = 5,
) -> Dict[str, float | np.ndarray]:
    """Calcule l'ensemble des métriques cliniques par classe et globales.

    Returns
    -------
    Un dictionnaire contenant :
      - ``sensitivity``  : array (n_classes,) — rappel par classe
      - ``specificity``  : array (n_classes,) — spécificité par classe
      - ``ppv``          : array (n_classes,) — VPP (précision) par classe
      - ``f1_per_class`` : array (n_classes,) — F1 par classe
      - ``macro_f1``     : float — F1 macro (recommandé pour datasets déséquilibrés)
      - ``accuracy``     : float
      - ``conf_matrix``  : ndarray (n_classes, n_classes)
      - ``report_str``   : str — rapport sklearn formaté
    """
    if class_names is None:
        class_names = [f"Classe {i}" for i in range(n_classes)]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    sensitivity = recall_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
    ppv         = precision_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
    f1_per_cls  = f1_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
    macro_f1    = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    accuracy    = float((y_true == y_pred).mean())

    # Spécificité par classe : TN / (TN + FP)
    # Pour chaque classe c : TN = tous les vrais négatifs dans le contexte one-vs-rest
    specificity = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificity[c] = tn / (tn + fp + 1e-8)

    report_str = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return {
        "sensitivity":  sensitivity,
        "specificity":  specificity,
        "ppv":          ppv,
        "f1_per_class": f1_per_cls,
        "macro_f1":     macro_f1,
        "accuracy":     accuracy,
        "conf_matrix":  cm,
        "report_str":   report_str,
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm:          np.ndarray,
    class_names: list,
    title:       str = "Matrice de confusion",
    save_path:   Optional[str | Path] = None,
) -> None:
    """Affiche la matrice de confusion normalisée et brute côte à côte.

    La normalisation par ligne (sur les vrais) permet de lire directement
    la sensibilité par classe, indépendamment du déséquilibre de données.
    """
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Comptages bruts", "Normalisée par ligne (Sensibilité)"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_xlabel("Prédiction", fontsize=11)
        ax.set_ylabel("Vérité terrain", fontsize=11)
        ax.set_title(subtitle, fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_metrics_by_class(
    metrics:     Dict,
    class_names: list,
    save_path:   Optional[str | Path] = None,
) -> None:
    """Graphique en barres groupées : Se, Sp, VPP, F1 par classe."""
    n_cls = len(class_names)
    x     = np.arange(n_cls)
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - 1.5*width, metrics["sensitivity"],  width, label="Sensibilité (Se)")
    ax.bar(x - 0.5*width, metrics["specificity"],  width, label="Spécificité (Sp)")
    ax.bar(x + 0.5*width, metrics["ppv"],          width, label="VPP (Précision)")
    ax.bar(x + 1.5*width, metrics["f1_per_class"], width, label="F1-score")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Métriques par classe  (Macro F1 = {metrics['macro_f1']:.3f})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_learning_curves(
    history:   Dict,
    model_name: str = "modèle",
    save_path:  Optional[str | Path] = None,
) -> None:
    """Trace les courbes d'apprentissage (loss et accuracy) train vs validation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Validation")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Courbe de perte — {model_name}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Validation")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Courbe d'exactitude — {model_name}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Comparaison de plusieurs modèles
# ---------------------------------------------------------------------------

def compare_models(results: Dict[str, Dict]) -> None:
    """Affiche un tableau récapitulatif des performances de plusieurs modèles.

    Parameters
    ----------
    results : dict
        Clé = nom du modèle, valeur = dict retourné par ``compute_metrics``.
    """
    print(f"\n{'Modèle':<20} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 42)
    for name, m in results.items():
        print(f"{name:<20} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")
    print()


# ---------------------------------------------------------------------------
# Interface en ligne de commande
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from src.data_processing import CLASS_NAMES
    from src.models          import build_model

    parser = argparse.ArgumentParser(description="Évaluer un modèle entraîné sur le jeu de test.")
    parser.add_argument("--model",     default="resnet1d", choices=["cnn1d", "resnet1d", "cnn_lstm"])
    parser.add_argument("--weights",   required=True, help="Chemin vers le fichier .pt")
    parser.add_argument("--test-data", required=True, help="Chemin vers le .npz contenant X_test/y_test")
    parser.add_argument("--save-dir",  default="results")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    with np.load(args.test_data) as f:
        X_test = f["X_test"]
        y_test = f["y_test"]

    y_pred, _ = predict(model, X_test)
    metrics   = compute_metrics(y_test, y_pred, class_names=CLASS_NAMES)

    print(metrics["report_str"])
    print(f"\nMacro F1 : {metrics['macro_f1']:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")

    plot_confusion_matrix(
        metrics["conf_matrix"], CLASS_NAMES,
        title=f"Matrice de confusion — {args.model}",
        save_path=save_dir / f"{args.model}_confusion_matrix.png",
    )
    plot_metrics_by_class(
        metrics, CLASS_NAMES,
        save_path=save_dir / f"{args.model}_metrics_by_class.png",
    )
