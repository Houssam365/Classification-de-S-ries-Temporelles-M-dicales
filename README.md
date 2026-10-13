# Classification de Séries Temporelles Médicales — ECG MIT-BIH

Classification automatique de signaux ECG pour détecter des arythmies cardiaques à partir
de la base MIT-BIH Arrhythmia.  L'objectif est double : **haute performance prédictive**
et **interprétabilité** des décisions pour validation clinique.

---

## Problématique

Le cœur produit un signal électrique périodique capturé par l'ECG.  Certaines anomalies
de ce signal — appelées **arythmies** — peuvent être bénignes ou mortelles.  Un système
de classification automatique doit :

- Identifier 5 types de battements selon la norme AAMI EC57 (N, S, V, F, Q)
- Maintenir une **sensibilité élevée** sur les classes critiques (V, S)
- Être **interprétable** pour permettre la validation par des cardiologues

---

## Structure du projet

```
├── data/                      # Données brutes MIT-BIH (téléchargées par setup_env.sh)
│   └── decoded/               # Signaux décodés en .npz (un fichier par enregistrement)
├── notebooks/
│   └── demo.ipynb             # Notebook pédagogique complet (point d'entrée recommandé)
├── results/                   # Modèles sauvegardés et figures générées
├── scripts/
│   └── download_data.py       # Téléchargement et décodage MIT-BIH via l'API PhysioNet
├── src/
│   ├── data_processing.py     # Segmentation, normalisation, découpage patient-level
│   ├── augmentation.py        # Jittering, time warping, magnitude warping, rééquilibrage
│   ├── models.py              # CNN1D, ResNet1D, CNN-LSTM + baseline handcrafted features
│   ├── train.py               # Boucle d'entraînement, early stopping, class weights
│   ├── evaluate.py            # Métriques médicales (Se, Sp, VPP, F1), visualisations
│   └── interpretability.py    # GradCAM 1D, attention weights, filtres convolutifs
├── requirements.txt
└── setup_env.sh               # Installation complète en une commande
```

---

## Installation et téléchargement des données

```bash
bash setup_env.sh
```

Ce script installe les dépendances Python et télécharge automatiquement la base MIT-BIH
depuis PhysioNet (public, pas d'authentification requise).

**Installation manuelle** :

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

---

## Démarrage rapide

Ouvrir le notebook `notebooks/demo.ipynb` (compatible Jupyter et Google Colab).

Il couvre dans l'ordre :

1. Chargement et exploration des données ECG
2. Visualisation des 5 classes d'arythmies
3. Baseline Random Forest avec features manuelles
4. Entraînement des modèles CNN1D, ResNet1D, CNN-LSTM
5. Évaluation clinique (sensibilité, spécificité, VPP, Macro F1)
6. Interprétabilité : GradCAM, attention, filtres convolutifs

---

## Utilisation en ligne de commande

**Entraînement :**

```bash
python -m src.train --model resnet1d --epochs 50 --save-dir results/
```

**Évaluation :**

```bash
python -m src.evaluate \
    --model resnet1d \
    --weights results/resnet1d_best.pt \
    --test-data results/resnet1d_test_data.npz \
    --save-dir results/
```

---

## Données

| Base | Enregistrements | Durée | Fréquence | Classes |
|------|-----------------|-------|-----------|---------|
| MIT-BIH Arrhythmia | 48 patients | 30 min chacun | 360 Hz | 5 (AAMI) |

**Classes AAMI** :

| Code | Description | Battements typiques |
|------|-------------|---------------------|
| N | Normal + BBB | ~80 % |
| S | Supra-ventriculaire ectopique | ~3 % |
| V | Ventriculaire ectopique | ~7 % |
| F | Fusion | ~0.5 % |
| Q | Inconnu / pacemaker | ~0.8 % |

Le fort déséquilibre de classes est géré par :
- Sur-échantillonnage augmenté des classes minoritaires
- Loss pondérée inversement proportionnelle à la fréquence de classe

---

## Architectures

| Modèle | Paramètres | Philosophie |
|--------|------------|-------------|
| `CNN1D` | ~200 K | Extraction de motifs locaux (morphologie QRS) |
| `ResNet1D` | ~600 K | Réseau profond avec skip connections, inspiration Wang et al. 2017 |
| `CNN-LSTM` | ~400 K | Local (CNN) + temporel long terme (LSTM) + attention de Bahdanau |

---

## Métriques d'évaluation

Conformément à la norme **AAMI EC57**, on rapporte pour chaque classe :

- **Sensibilité (Se)** = TP / (TP + FN) — ne pas manquer une arythmie
- **Spécificité (Sp)** = TN / (TN + FP) — éviter les fausses alarmes
- **VPP (Précision)** = TP / (TP + FP) — fiabilité de l'alerte
- **Macro F1** — métrique globale équitable pour données déséquilibrées

---

## Reproductibilité

- Stratification **par enregistrement** (patient-level) : jamais de battements du même
  patient simultanément en train et test
- Graines aléatoires fixées partout (`seed=42`)
- `torch.backends.cudnn.deterministic = True`

---

## Références

- Fawaz et al. (2019) — *Deep learning for time series classification: a review*
  [arXiv:1809.04356](https://arxiv.org/abs/1809.04356)
- Wang et al. (2017) — *Time series classification from scratch with deep neural networks*
  [arXiv:1611.06455](https://arxiv.org/abs/1611.06455)
- MIT-BIH Arrhythmia Database : [physionet.org/content/mitdb/1.0.0](https://physionet.org/content/mitdb/1.0.0/)
