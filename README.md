# Classification-de-S-ries-Temporelles-M-dicales
Classifier automatiquement des signaux ECG pour détecter différents types d'arythmies cardiaques. L'enjeu est double : performance prédictive élevée (contexte médical) et interprétabilité des décisions pour validation clinique.  Tâche attendue : Classification multi-classes de séries temporelles

Pour installer les dépendances du projet et télécharger la base MIT-BIH Arrhythmia depuis PhysioNet directement dans le dossier `data/`, exécutez depuis la racine du projet :

```bash
bash setup_env.sh
```

Le script installe d'abord les dépendances listées dans `requirements.txt`, puis reprend les téléchargements interrompus et ne retélécharge pas les fichiers inchangés.

Depuis Python ou le notebook Colab `notebooks/demo.ipynb`, le téléchargement peut aussi être lancé avec l'API WFDB de PhysioNet :

```python
from scripts.download_data import download_mitdb

data_dir = download_mitdb()
```

Cette fonction enregistre les enregistrements et annotations dans `data/`. Installez les dépendances avec `pip install -r requirements.txt` si nécessaire.

L'exécution de `python scripts/download_data.py` télécharge également les signaux et annotations décodés dans `data/decoded/`, avec un fichier `.npz` par enregistrement. Pour lire un enregistrement :

```python
import numpy as np

with np.load("data/decoded/100.npz") as record:
	signal = record["signal"]
	sampling_frequency = float(record["sampling_frequency"])
	lead_names = record["lead_names"]
	annotation_samples = record["annotation_samples"]
	annotation_symbols = record["annotation_symbols"]
```
