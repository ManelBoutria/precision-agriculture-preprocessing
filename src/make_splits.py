"""make_splits.py : génère le test verrouillé et les plis de validation.

Usage : python src/make_splits.py chemin/vers/dataset.csv
Sorties (dans results/splits/) :
    dev_index.csv, test_index.csv, splits_validation.csv, splits_temporel.csv
Les colonnes "index" contiennent les index de lignes du CSV ORIGINAL
(utiliser df.loc[index] après avoir lu le même fichier, sans le trier ni le filtrer).
"""
import os
import sys

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, TimeSeriesSplit

SEED = 42             # graine globale du projet
FRACTION_TEST = 0.2   # 20 % des dernières lignes (chronologiques) pour le test
N_PLIS = 10           # validation croisée stratifiée : 10 plis
N_REPETITIONS = 3     # répétée 3 fois (3 x 10 = 30 scores par modèle)
N_PLIS_TEMPORELS = 5  # TimeSeriesSplit : 5 plis
GAP = 48              # 48 pas de 30 min = 24 h entre entraînement et validation
CIBLE = "environmental_condition"
SORTIE = "results/splits"


def charger(chemin):
    """Lit le CSV, convertit la date et vérifie l'ordre chronologique."""
    df = pd.read_csv(chemin)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    assert df["timestamp"].is_monotonic_increasing, "Les données ne sont pas triées par date"
    return df


def separer_test(df):
    """Le test = les dernières lignes (le df est trié chronologiquement)."""
    n = len(df)
    n_test = int(n * FRACTION_TEST)
    dev = df.iloc[: n - n_test]
    test = df.iloc[n - n_test:]
    return dev, test


def plis_stratifies(dev):
    """3 répétitions x 10 plis stratifiés sur le développement.
    On enregistre uniquement les lignes de validation de chaque pli ;
    l'entraînement = tout le développement privé de ces lignes."""
    y_dev = dev[CIBLE]
    rskf = RepeatedStratifiedKFold(
        n_splits=N_PLIS, n_repeats=N_REPETITIONS, random_state=SEED
    )
    lignes = []
    for i, (_, idx_val) in enumerate(rskf.split(dev, y_dev)):
        repetition = i // N_PLIS
        pli = i % N_PLIS
        for idx in dev.index[idx_val]:
            lignes.append({"repetition": repetition, "fold": pli, "index": idx})
    return pd.DataFrame(lignes)


def plis_temporels(dev):
    """TimeSeriesSplit : on s'entraîne sur le passé, on valide sur la période suivante.
    Ici l'entraînement n'est pas « le reste » (à cause du gap), donc on l'enregistre."""
    tscv = TimeSeriesSplit(n_splits=N_PLIS_TEMPORELS, gap=GAP)
    lignes = []
    for pli, (idx_train, idx_val) in enumerate(tscv.split(dev)):
        for idx in dev.index[idx_train]:
            lignes.append({"fold": pli, "ensemble": "train", "index": idx})
        for idx in dev.index[idx_val]:
            lignes.append({"fold": pli, "ensemble": "val", "index": idx})
    return pd.DataFrame(lignes)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage : python src/make_splits.py chemin/vers/dataset.csv")

    os.makedirs(SORTIE, exist_ok=True)
    df = charger(sys.argv[1])
    dev, test = separer_test(df)

    splits = plis_stratifies(dev)
    splits_ts = plis_temporels(dev)

    # Vérifications de cohérence (le script s'arrête si quelque chose ne va pas)
    assert len(set(dev.index) & set(test.index)) == 0, "Fuite : dev et test se chevauchent"
    assert len(splits) == len(dev) * N_REPETITIONS, "Nombre de lignes de validation inattendu"

    pd.DataFrame({"index": dev.index}).to_csv(f"{SORTIE}/dev_index.csv", index=False)
    pd.DataFrame({"index": test.index}).to_csv(f"{SORTIE}/test_index.csv", index=False)
    splits.to_csv(f"{SORTIE}/splits_validation.csv", index=False)
    splits_ts.to_csv(f"{SORTIE}/splits_temporel.csv", index=False)

    print(f"Dev : {len(dev)} lignes | Test : {len(test)} lignes")
    print(f"Plis stratifiés : {len(splits)} lignes de validation")
    print(f"Plis temporels : {len(splits_ts)} lignes")
    print(f"Fichiers écrits dans {SORTIE}/")
