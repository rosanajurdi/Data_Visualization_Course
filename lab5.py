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
#from statsmodels.graphics.tsaplots import plot_acf

# Configuration du chemin de données
DATA_PATH = Path(".")  # On suppose que le dataset est dans le même répertoire

# Chargement du dataset
passenger_df = pd.read_csv("./datasets/AirPassengersDates.csv")

# Affichage des premières lignes
print(passenger_df.head())

# Affichage des informations sur le DataFrame
print(passenger_df.info())

# Section 2: Manipulation des dates
# --------------------------------

# Conversion de la colonne Date en datetime
passenger_df["Date"] = pd.to_datetime(passenger_df["Date"])

# Vérification de la conversion
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
sns.barplot(x="Month", y="#Passengers", data=passengers_per_month)
plt.title("Total Passengers per Month")
plt.xlabel("Month")
plt.ylabel("Total Passengers")
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
passenger_df.set_index("Date", inplace=True)

# 1. Sur-échantillonnage (upsampling)
# Rééchantillonnage à fréquence quotidienne
daily_passengers = passenger_df.resample('D').asfreq()

# Interpolation des valeurs manquantes
daily_passengers['#Passengers'] = daily_passengers['#Passengers'].interpolate(method='linear')

plt.figure(figsize=(12, 6))
daily_passengers['#Passengers'].plot(label='Upsampled and Interpolated', linestyle='--')
passenger_df['#Passengers'].plot(label='Original', alpha=0.7)  # Tracé des données originales pour comparaison
plt.title('Upsampling to Daily Frequency')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.show()

# 2. Sous-échantillonnage (downsampling)
# Rééchantillonnage à fréquence annuelle
yearly_passengers = passenger_df.resample("Y")["#Passengers"].mean()

plt.figure(figsize=(12, 6))
yearly_passengers.plot(marker="o", label="Yearly Average")
passenger_df["#Passengers"].plot(alpha=0.5, label="Original")  # Données originales
plt.title("Downsampling to Yearly Frequency")
plt.xlabel("Year")
plt.ylabel("Average Passengers")
plt.legend()
plt.show()

# Section 6: Shift et tshift (Analyse de décalage)
# -----------------------------------------------

# 1. Utilisation de `shift()` pour l'analyse de décalage
# Reset de l'index pour utiliser Date comme colonne
# passenger_df.reset_index(inplace=False)  # Cette ligne n'a pas d'effet car inplace=False

# Création des colonnes décalées
passenger_df["#Passengers_Shift"] = passenger_df["#Passengers"].shift(periods=1)  # Décalage des données de 1 période
passenger_df["#Passengers_tShift"] = passenger_df["#Passengers"].shift(periods=1, freq="MS")  # Décalage de l'index de 1 début de mois

print(passenger_df.head())

# Visualisation
plt.figure(figsize=(12, 6))
passenger_df["#Passengers"].plot(label="Original")
passenger_df["#Passengers_Shift"].plot(label="Shifted")
passenger_df["#Passengers_tShift"].plot(label="tShifted")
plt.title("Shift vs tShift")
plt.legend()
plt.show()

# Section 7: Autocorrélation
# -------------------------

# 1. Tracé d'autocorrélation
plt.figure(figsize=(10, 5))
plot_acf(passenger_df["#Passengers"], lags=30, ax=plt.gca(), title="Autocorrelation Function (ACF)")  # Utilisation des décalages jusqu'à 30
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()