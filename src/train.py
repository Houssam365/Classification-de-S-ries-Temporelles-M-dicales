"""
Boucle d'entraînement pour les modèles de classification ECG.

Philosophie du module
---------------------
L'entraînement d'un classifieur en contexte médical requiert des précautions
spécifiques par rapport à un problème générique :

  1. **Gestion du déséquilibre de classes** : on utilise une ``CrossEntropyLoss``
     pondérée (``class_weight``), donnant plus d'importance aux arythmies rares
     mais critiques (V, S, F) lors du calcul du gradient.

  2. **Validation per-epoch** : on surveille la perte de validation à chaque époque
     et on sauvegarde le meilleur modèle (early stopping).  Cela évite le surapprentissage
     sur les classes majoritaires.

  3. **Learning rate scheduling** : un ``CosineAnnealingLR`` réduit progressivement
     le taux d'apprentissage, affinant la convergence en fin d'entraînement.

  4. **Reproductibilité** : les graines NumPy, Python et PyTorch sont fixées avant
     l'initialisation du modèle et du DataLoader.

Ce module expose la classe ``Trainer`` qui encapsule toute la logique d'entraînement
et peut être importée dans le notebook ou lancée en ligne de commande.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Graine globale de reproductibilité
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fixe les graines de tous les générateurs aléatoires utilisés."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Construction des DataLoaders
# ---------------------------------------------------------------------------

def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """Transforme les tableaux NumPy en DataLoaders PyTorch.

    Les signaux sont mis en forme (B, 1, T) pour correspondre à l'entrée des
    réseaux convolutifs 1D (Conv1d attend un canal explicite).
    """

    def _make_tensor_dataset(X, y):
        # Ajout de la dimension canal : (N, T) → (N, 1, T)
        X_t = torch.from_numpy(X).unsqueeze(1).float()
        y_t = torch.from_numpy(y).long()
        return TensorDataset(X_t, y_t)

    train_ds = _make_tensor_dataset(X_train, y_train)
    val_ds   = _make_tensor_dataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Calcul des poids de classes pour la loss pondérée
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray, n_classes: int = 5) -> torch.Tensor:
    """Calcule des poids inversement proportionnels à la fréquence de classe.

    Formule : w_c = N_total / (n_classes × N_c)

    Les classes rares reçoivent un poids élevé → le gradient les pénalise plus
    lors des erreurs de classification.  En contexte médical, rater une arythmie
    ventriculaire critique (V) est bien plus grave que sur-détecter un battement normal.
    """
    counts  = np.bincount(y, minlength=n_classes).astype(np.float32)
    total   = counts.sum()
    weights = total / (n_classes * (counts + 1e-8))
    return torch.from_numpy(weights).float()


# ---------------------------------------------------------------------------
# Classe Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Encapsule la boucle d'entraînement, de validation et la sauvegarde du modèle.

    Paramètres
    ----------
    model : nn.Module
        Modèle à entraîner (CNN1D, ResNet1D ou CNN_LSTM).
    n_classes : int
        Nombre de classes de sortie.
    device : str | torch.device
        'cuda' ou 'cpu'.
    lr : float
        Taux d'apprentissage initial.
    weight_decay : float
        Régularisation L2 de l'optimiseur Adam.
    epochs : int
        Nombre maximal d'époques.
    patience : int
        Nombre d'époques sans amélioration avant arrêt anticipé (early stopping).
    save_dir : str | Path
        Répertoire où sauvegarder le meilleur modèle.
    """

    def __init__(
        self,
        model:        nn.Module,
        n_classes:    int = 5,
        device:       str = "auto",
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        epochs:       int = 50,
        patience:     int = 10,
        save_dir:     str | Path = "results",
    ) -> None:
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.model       = model.to(self.device)
        self.n_classes   = n_classes
        self.epochs      = epochs
        self.patience    = patience
        self.save_dir    = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer   = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        # Historique pour les courbes d'apprentissage
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [],  "val_acc": [],
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        y_train:      Optional[np.ndarray] = None,
        model_name:   str = "model",
    ) -> Dict[str, list]:
        """Lance l'entraînement complet avec early stopping.

        Parameters
        ----------
        train_loader, val_loader : DataLoader
        y_train : ndarray optionnel
            Si fourni, calcule les poids de classes pour la loss pondérée.
        model_name : str
            Préfixe du fichier de sauvegarde.

        Returns
        -------
        history : dict contenant les courbes train/val loss et accuracy.
        """
        # Loss pondérée pour gérer le déséquilibre de classes
        if y_train is not None:
            weights   = compute_class_weights(y_train, self.n_classes).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        best_val_loss   = float("inf")
        epochs_no_impro = 0
        best_path       = self.save_dir / f"{model_name}_best.pt"

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(train_loader, criterion, training=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   criterion, training=False)

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            elapsed = time.time() - t0
            print(
                f"Époque {epoch:>3}/{self.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                f"({elapsed:.1f}s)"
            )

            # Sauvegarde du meilleur modèle (critère : val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_impro = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                epochs_no_impro += 1

            # Early stopping
            if epochs_no_impro >= self.patience:
                print(f"\nEarly stopping à l'époque {epoch} (patience={self.patience}).")
                break

        # Rechargement du meilleur modèle avant de rendre la main
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        print(f"\nMeilleur modèle rechargé depuis {best_path}.")
        return self.history

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        training: bool,
    ) -> Tuple[float, float]:
        """Exécute une époque complète (train ou évaluation).

        Returns
        -------
        avg_loss : float
        accuracy : float
        """
        self.model.train(training)
        total_loss, total_correct, total_samples = 0.0, 0, 0

        with torch.set_grad_enabled(training):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Gestion du retour optionnel (CNN_LSTM retourne aussi les poids d'attention)
                out = self.model(X_batch)
                logits = out[0] if isinstance(out, (tuple, list)) else out

                loss = criterion(logits, y_batch)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping pour stabiliser l'entraînement du LSTM
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()

                total_loss    += loss.item() * len(y_batch)
                total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total_samples += len(y_batch)

        avg_loss = total_loss    / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Interface en ligne de commande
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from src.data_processing import ECGDataset
    from src.augmentation    import oversample_minority_classes
    from src.models          import build_model

    parser = argparse.ArgumentParser(description="Entraîner un modèle de classification ECG.")
    parser.add_argument("--model",      default="resnet1d", choices=["cnn1d", "resnet1d", "cnn_lstm"])
    parser.add_argument("--decoded-dir", default="data/decoded")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--save-dir",   default="results")
    args = parser.parse_args()

    set_seed(args.seed)

    print("Chargement et segmentation des données…")
    dataset = ECGDataset(args.decoded_dir, seed=args.seed).load()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.split()
    print(f"  Train : {len(y_train)} beats | Val : {len(y_val)} | Test : {len(y_test)}")

    print("Rééquilibrage des classes minoritaires…")
    X_train, y_train = oversample_minority_classes(X_train, y_train, seed=args.seed)
    print(f"  Train après sur-échantillonnage : {len(y_train)} beats")

    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, args.batch_size)

    print(f"Construction du modèle '{args.model}'…")
    model = build_model(args.model)

    trainer = Trainer(model, epochs=args.epochs, lr=args.lr, save_dir=args.save_dir)
    history = trainer.fit(train_loader, val_loader, y_train, model_name=args.model)

    print("\nEntraînement terminé.")
    # Sauvegarde des données test pour l'évaluation ultérieure
    np.savez(
        Path(args.save_dir) / f"{args.model}_test_data.npz",
        X_test=X_test, y_test=y_test,
    )
    print(f"Données de test sauvegardées dans {args.save_dir}/{args.model}_test_data.npz")
