
"""temporal_features.py : features temporelles (étudiante 3, semaine 2).

Usage : python src/temporal_features.py chemin/vers/coded_data.csv
Sortie : results/temporal_features.csv (une ligne par ligne du CSV d'origine)

Les colonnes sont indexées par l'INDEX DE LIGNE d'origine (le même que dans
results/splits/*.csv) : il suffit de joindre sur l'index.

Choix justifiés par l'analyse temporelle (voir Exploratory_temporal.csv) :
- heure_sin / heure_cos : cycle journalier net de la température (ACF).
- lag1 et lag2 de la température de l'air et de l'humidité relative : PACF fort aux lags 1-2.
- Rien pour l'humidité du sol, la pression, les nutriments (NPK), etc. : pas de dépendance
  temporelle (Ljung-Box p > 0,05 ; azote p = 0,025 jugé fortuit : PACF ~ 0 et 20 tests réalisés).
Aucune feature n'utilise la cible, et aucune n'utilise le futur (shift positif uniquement).
Les lags sont calculés sur la série GLOBALE triée par date (pas pas de 30 min), car les
métadonnées capteur/zone ne sont pas fiables (limite à documenter dans le rapport).
"""
import os
import sys

import numpy as np
import pandas as pd

COL_TEMPS = "timestamp"
# Variables avec dépendance temporelle ET mémoire courte (PACF fort aux lags 1-2) qui pilotent
# les classes Heat_Stress / High_Humidity. Les autres variables dépendantes (température du sol,
# rayonnement, lumière, évapotranspiration, leaf_wetness) sont très corrélées à celles-ci ;
# elles restent des candidats pour la sélection de variables (étudiant 4).
LAG_VARIABLES = ["air_temperature_c", "relative_humidity_pct"]
LAGS = [1, 2]                       # en pas de 30 minutes
PAS = pd.Timedelta(minutes=30)
SORTIE = "results/temporal_features.csv"


def charger(chemin):
    """Lit le CSV en gardant l'index de ligne d'origine."""
    df = pd.read_csv(chemin)
    if "Unnamed: 0" in df.columns:          # coded_data.csv contient l'index d'origine
        df = df.set_index("Unnamed: 0")
        df.index.name = None
    df[COL_TEMPS] = pd.to_datetime(df[COL_TEMPS])
    return df


def ajouter_features_temporelles(df):
    """Retourne un DataFrame avec uniquement les nouvelles colonnes (même index que df)."""
    # Garde-fous : les lags n'ont de sens que si la série est triée et régulière
    assert df[COL_TEMPS].is_monotonic_increasing, "Données non triées par date"
    assert (df[COL_TEMPS].diff().dropna() == PAS).all(), "Pas de temps non régulier (30 min)"

    out = pd.DataFrame(index=df.index)

    # Encodage cyclique de l'heure : 23h30 et 0h00 deviennent proches
    heure = df[COL_TEMPS].dt.hour + df[COL_TEMPS].dt.minute / 60
    out["heure_sin"] = np.sin(2 * np.pi * heure / 24)
    out["heure_cos"] = np.cos(2 * np.pi * heure / 24)

    # Lags : valeur observée k pas AVANT (shift positif = passé uniquement)
    for var in LAG_VARIABLES:
        for k in LAGS:
            out[f"{var}_lag{k}"] = df[var].shift(k)

    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage : python src/temporal_features.py chemin/vers/coded_data.csv")

    df = charger(sys.argv[1])
    feats = ajouter_features_temporelles(df)

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    feats.to_csv(SORTIE, index_label="index", encoding="utf-8")

    print(f"{feats.shape[1]} features créées pour {len(feats)} lignes")
    print("Valeurs manquantes (début de série) :")
    print(feats.isna().sum()[feats.isna().sum() > 0])
    print(f"Fichier écrit : {SORTIE}")
