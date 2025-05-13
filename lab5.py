# Analyse et visualisation de séries temporelles avec Pandas
# =========================================================

# Section 1: Configuration et chargement des données
# -------------------------------------------------

# Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration du chemin de données
DATA_PATH = Path(".")

# Création du dataset (puisque le fichier original n'est pas disponible)
# Nous allons créer des données similaires au jeu de données AirPassengers
date_rng = pd.date_range(start='1949-01-01', end='1960-12-31', freq='M')
np.random.seed(42)  # Pour la reproductibilité

# Création d'une tendance et d'une saisonnalité
trend = np.linspace(100, 600, len(date_rng))  # Tendance croissante de 100 à 600
seasonality = 50 * np.sin(np.arange(len(date_rng)) * (2 * np.pi / 12))  # Cycle saisonnier de 12 mois
random = np.random.normal(0, 20, len(date_rng))  # Composante aléatoire

# Combiner les composantes
passengers = trend + seasonality + random
passengers = np.round(passengers).astype(int)  # Conversion en entiers (nombres de passagers)

# Création du DataFrame
passenger_df = pd.DataFrame({
    'Date': date_rng,
    '#Passengers': passengers
})

# Affichage des premières lignes
print(passenger_df.head())

# Affichage des informations sur le DataFrame
print(passenger_df.info())

# Section 2: Manipulation des dates
# --------------------------------

# Les dates sont déjà au format datetime, donc pas besoin de les convertir
# Vérification du type de données
print(passenger_df.info())

# Extraction des composants de date
passenger_df["Month"] = passenger_df["Date"].dt.month
passenger_df["Day"] = passenger_df["Date"].dt.day
passenger_df["Day_Name"] = passenger_df["Date"].dt.day_name()

print(passenger_df.head())

# Section 3: Visualisation et analyse des séries temporelles
# ---------------------------------------------------------

# 1. Tracé de série temporelle basique
plt.figure(figsize=(12, 6))
plt.plot(passenger_df["Date"], passenger_df["#Passengers"])
plt.title("Air Passengers Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.grid(True)
plt.show()

# 2. Agrégation et diagramme en barres
passengers_per_month = (
    passenger_df.groupby("Month")["#Passengers"].sum().reset_index()
)

plt.figure(figsize=(10, 5))

# Création d'une palette de couleurs avec 12 couleurs différentes pour les 12 mois
colors = plt.cm.tab20(np.linspace(0, 1, 12))

# Utilisation de plt.bar au lieu de sns.barplot pour un contrôle personnalisé des couleurs
bars = plt.bar(passengers_per_month["Month"], passengers_per_month["#Passengers"], color=colors)

# Ajout d'étiquettes et de titres
plt.title("Total Passengers per Month", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Total Passengers", fontsize=12)

# Ajustement des ticks sur l'axe x pour montrer tous les mois
plt.xticks(range(1, 13))

# Ajout d'une grille pour faciliter la lecture
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 3. Moyenne et écart-type
mean_passengers = passenger_df["#Passengers"].mean()
std_passengers = passenger_df["#Passengers"].std()

plt.figure(figsize=(12, 6))
plt.plot(passenger_df["Date"], passenger_df["#Passengers"], label="Passengers")
plt.axhline(
    mean_passengers, color="r", linestyle="--", label=f"Mean: {mean_passengers:.2f}"
)
plt.axhline(
    mean_passengers + std_passengers,
    color="g",
    linestyle="--",
    label=f"Mean + Std: {mean_passengers + std_passengers:.2f}",
)
plt.axhline(
    mean_passengers - std_passengers,
    color="g",
    linestyle="--",
    label=f"Mean - Std: {mean_passengers - std_passengers:.2f}",
)
plt.title("Passengers with Mean and Standard Deviation")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.legend()
plt.grid(True)
plt.show()

# Section 4: Détection des valeurs aberrantes
# -------------------------------------------

# 1. Calcul du Z-Score
passenger_df["Z-Score"] = (
    passenger_df["#Passengers"] - passenger_df["#Passengers"].mean()
) / passenger_df["#Passengers"].std()
passenger_df["Absolute_Z-Score"] = abs(passenger_df["Z-Score"])

print(passenger_df.sort_values(by="Absolute_Z-Score", ascending=False).head(10))

# 2. Visualisation des valeurs aberrantes
outliers = passenger_df[(passenger_df["Absolute_Z-Score"] > 2)]  # Définition des valeurs aberrantes

plt.figure(figsize=(12, 6))
plt.plot(passenger_df["Date"], passenger_df["#Passengers"], label="Passengers")
plt.scatter(outliers["Date"], outliers["#Passengers"], color="red", label="Outliers")
plt.title("Air Passengers with Outliers")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.legend()
plt.grid(True)
plt.show()

# Section 5: Rééchantillonnage
# ---------------------------

# Configuration de 'Date' comme index pour le rééchantillonnage
passenger_df_indexed = passenger_df.set_index("Date")

# 1. Sur-échantillonnage (upsampling)
# Rééchantillonnage à fréquence quotidienne
daily_passengers = passenger_df_indexed.resample('D').asfreq()

# Interpolation des valeurs manquantes
daily_passengers['#Passengers'] = daily_passengers['#Passengers'].interpolate(method='linear')

plt.figure(figsize=(12, 6))
daily_passengers['#Passengers'].plot(label='Upsampled and Interpolated', linestyle='--')
passenger_df_indexed['#Passengers'].plot(label='Original', alpha=0.7)  # Tracé des données originales pour comparaison
plt.title('Upsampling to Daily Frequency')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.show()

# 2. Sous-échantillonnage (downsampling)
# Rééchantillonnage à fréquence annuelle
yearly_passengers = passenger_df_indexed.resample("Y")["#Passengers"].mean()

plt.figure(figsize=(12, 6))
yearly_passengers.plot(marker="o", label="Yearly Average")
passenger_df_indexed["#Passengers"].plot(alpha=0.5, label="Original")  # Données originales
plt.title("Downsampling to Yearly Frequency")
plt.xlabel("Year")
plt.ylabel("Average Passengers")
plt.legend()
plt.show()

# Section 6: Shift et tshift (Analyse de décalage)
# -----------------------------------------------

# 1. Utilisation de `shift()` pour l'analyse de décalage
# Création des colonnes décalées sur les données indexées
passenger_df_indexed["#Passengers_Shift"] = passenger_df_indexed["#Passengers"].shift(periods=1)  # Décalage des données de 1 période
passenger_df_indexed["#Passengers_tShift"] = passenger_df_indexed["#Passengers"].shift(periods=1, freq="MS")  # Décalage de l'index de 1 début de mois

print(passenger_df_indexed.head())

# Visualisation
plt.figure(figsize=(12, 6))
passenger_df_indexed["#Passengers"].plot(label="Original")
passenger_df_indexed["#Passengers_Shift"].plot(label="Shifted")
passenger_df_indexed["#Passengers_tShift"].plot(label="tShifted")
plt.title("Shift vs tShift")
plt.legend()
plt.show()

# Section 7: Autocorrélation
# -------------------------

# 1. Implémentation manuelle de l'autocorrélation
plt.figure(figsize=(10, 5))

# Calcul manuel de l'autocorrélation jusqu'à 30 lags
lags = 30
autocorr = []
series = passenger_df_indexed["#Passengers"]

for lag in range(lags + 1):
    # Pour le lag 0, la corrélation est toujours 1
    if lag == 0:
        autocorr.append(1)
    else:
        # Calcul de l'autocorrélation pour chaque lag
        correlation = series.autocorr(lag=lag)
        autocorr.append(correlation)

# Création du graphique d'autocorrélation
plt.bar(range(len(autocorr)), autocorr, width=0.3)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-0.2, linestyle=':', color='red', alpha=0.3)  # Ligne de significativité approx.
plt.axhline(y=0.2, linestyle=':', color='red', alpha=0.3)   # Ligne de significativité approx.
plt.ylim([-1, 1])
plt.title("Autocorrelation Function (ACF)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()