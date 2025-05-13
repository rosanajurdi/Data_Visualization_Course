import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# URL alternative pour le dataset olympique
url = "https://raw.githubusercontent.com/rgriff23/Olympic_history/master/data/athlete_events.csv"

try:
    # Téléchargement du dataset
    df = pd.read_csv(url)
    print(f"Données chargées avec succès: {df.shape[0]} lignes et {df.shape[1]} colonnes")
except Exception as e:
    print(f"Erreur lors du chargement des données: {e}")
    
    # Solution alternative si l'URL ne fonctionne pas
    print("Tentative avec une URL alternative...")
    url_alt = "https://raw.githubusercontent.com/heesoo4199/Olympic_history/master/data/athlete_events.csv"
    try:
        df = pd.read_csv(url_alt)
        print(f"Données chargées avec succès depuis l'URL alternative: {df.shape[0]} lignes et {df.shape[1]} colonnes")
    except Exception as e:
        print(f"Erreur avec l'URL alternative aussi: {e}")
        # Création d'un petit exemple de données pour démontrer le code
        print("Création d'un exemple de données pour démontrer le code...")
        # Création de données d'exemple pour permettre l'exécution du code
        sports = ['Athletics', 'Swimming', 'Cycling', 'Wrestling', 'Judo', 'Gymnastics', 'Basketball']
        medals = ['Gold', 'Silver', 'Bronze']
        nocs = ['USA', 'CHN', 'GBR', 'RUS', 'JPN', 'FRA', 'GER', 'AUS', 'ITA', 'CAN']
        
        np.random.seed(42)
        n = 1000
        data = {
            'ID': range(1, n+1),
            'Name': [f'Athlete_{i}' for i in range(1, n+1)],
            'Sex': np.random.choice(['M', 'F'], size=n),
            'Age': np.random.normal(25, 5, n).round().astype(int),
            'Height': np.random.normal(175, 15, n).round().astype(int),
            'Weight': np.random.normal(70, 15, n).round().astype(int),
            'Team': np.random.choice(nocs, size=n),
            'NOC': np.random.choice(nocs, size=n),
            'Year': [2016] * n,
            'Season': ['Summer'] * n,
            'City': ['Rio'] * n,
            'Sport': np.random.choice(sports, size=n, p=[0.25, 0.25, 0.15, 0.15, 0.10, 0.05, 0.05]),
            'Event': [f'Event_{i % 50}' for i in range(n)],
            'Medal': np.random.choice([*medals, np.nan], size=n, p=[0.1, 0.1, 0.1, 0.7])
        }
        df = pd.DataFrame(data)

# Examiner les premières lignes du DataFrame pour comprendre sa structure
print("\nAperçu des données olympiques:")
print(df.head())

# Vérifier les informations sur le DataFrame
print("\nInformations sur le DataFrame:")
print(df.info())

# 1. Filtrer le DataFrame pour inclure uniquement les médaillés de 2016
df_2016 = df[(df['Year'] == 2016) & (~df['Medal'].isna())]
print(f"\nNombre de médailles attribuées en 2016: {len(df_2016)}")

# 2. Compter le nombre de médailles par sport en 2016
medals_by_sport = df_2016['Sport'].value_counts()
print("\nNombre de médailles par sport en 2016:")
print(medals_by_sport.head(10))

# 3. Identifier les 5 premiers sports avec le plus grand nombre de médailles
top5_sports = medals_by_sport.head(5)
print("\nTop 5 des sports avec le plus grand nombre de médailles en 2016:")
print(top5_sports)

# 4. Filtrer pour ne garder que les données des 5 premiers sports
df_top5 = df_2016[df_2016['Sport'].isin(top5_sports.index)]
print(f"\nNombre de médailles dans les 5 premiers sports: {len(df_top5)}")

# 5. Créer un graphique à barres montrant le nombre de médailles par sport (top 5)
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=top5_sports.index, y=top5_sports.values, palette='viridis')
plt.title('Nombre de médailles attribuées dans les 5 premiers sports (2016)', fontsize=15)
plt.xlabel('Sport', fontsize=12)
plt.ylabel('Nombre de médailles', fontsize=12)
plt.xticks(rotation=45)
for i, v in enumerate(top5_sports.values):
    plt.text(i, v + 5, str(v), ha='center')
plt.tight_layout()
plt.show()

# 6. Distribution de l'âge des médaillés dans les 5 premiers sports
plt.figure(figsize=(14, 7))
sns.histplot(data=df_top5, x='Age', hue='Sport', bins=20, multiple='stack', palette='viridis')
plt.title('Distribution de l\'âge des médaillés dans les 5 premiers sports (2016)', fontsize=15)
plt.xlabel('Âge', fontsize=12)
plt.ylabel('Nombre d\'athlètes', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Version alternative avec des boxplots pour mieux comparer les distributions d'âge par sport
plt.figure(figsize=(14, 7))
sns.boxplot(data=df_top5, x='Sport', y='Age', palette='viridis')
plt.title('Distribution de l\'âge des médaillés par sport (2016)', fontsize=15)
plt.xlabel('Sport', fontsize=12)
plt.ylabel('Âge', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 7. Équipes nationales ayant remporté le plus grand nombre de médailles dans les 5 premiers sports
medals_by_country = df_top5['NOC'].value_counts().head(15)
plt.figure(figsize=(14, 8))
sns.barplot(x=medals_by_country.index, y=medals_by_country.values, palette='mako')
plt.title('Top 15 des pays par nombre de médailles dans les 5 premiers sports (2016)', fontsize=15)
plt.xlabel('Pays (code NOC)', fontsize=12)
plt.ylabel('Nombre de médailles', fontsize=12)
plt.xticks(rotation=45)
for i, v in enumerate(medals_by_country.values):
    plt.text(i, v + 1, str(v), ha='center')
plt.tight_layout()
plt.show()

# 8. Poids moyen des athlètes médaillés par genre dans les 5 premiers sports
# Filtrer les données pour exclure les valeurs NaN pour Weight
df_weight = df_top5.dropna(subset=['Weight'])

# Calculer le poids moyen par sport et par sexe
avg_weight = df_weight.groupby(['Sport', 'Sex'])['Weight'].mean().unstack()
print("\nPoids moyen des athlètes par sport et par sexe:")
print(avg_weight)

# Créer un graphique à barres groupées
if not avg_weight.empty:
    avg_weight.plot(kind='bar', figsize=(14, 8))
    plt.title('Poids moyen des médaillés par sport et par sexe (2016)', fontsize=15)
    plt.xlabel('Sport', fontsize=12)
    plt.ylabel('Poids moyen (kg)', fontsize=12)
    plt.legend(title='Sexe')
    plt.grid(axis='y', alpha=0.3)
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.1f')
    plt.tight_layout()
    plt.show()
else:
    print("Données insuffisantes pour créer le graphique de poids moyen")

# Analyse plus approfondie: répartition des médailles (Or, Argent, Bronze) par sport
plt.figure(figsize=(14, 8))
medal_counts = df_top5.groupby(['Sport', 'Medal']).size().unstack()
if not medal_counts.empty:
    medal_counts.plot(kind='bar', stacked=False, figsize=(14, 8))
    plt.title('Répartition des types de médailles par sport (2016)', fontsize=15)
    plt.xlabel('Sport', fontsize=12)
    plt.ylabel('Nombre de médailles', fontsize=12)
    plt.legend(title='Type de médaille')
    plt.grid(axis='y', alpha=0.3)
    for container in plt.gca().containers:
        plt.gca().bar_label(container)
    plt.tight_layout()
    plt.show()
else:
    print("Données insuffisantes pour créer le graphique de répartition des médailles")

# Bonus: Analyse de la taille des athlètes par sport
plt.figure(figsize=(14, 8))
df_height = df_top5.dropna(subset=['Height'])
if not df_height.empty:
    sns.boxplot(data=df_height, x='Sport', y='Height', hue='Sex', palette='Set2')
    plt.title('Taille des médaiillés par sport et par sexe (2016)', fontsize=15)
    plt.xlabel('Sport', fontsize=12)
    plt.ylabel('Taille (cm)', fontsize=12)
    plt.legend(title='Sexe')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("Données insuffisantes pour créer le graphique de taille des athlètes")