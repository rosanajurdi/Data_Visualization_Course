import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
from plotly.offline import iplot
import numpy as np

# Chargement des données
url = "https://raw.githubusercontent.com/TrainingByPackt/Interactive-Data-Visualization-with-Python/master/datasets/diamonds.csv"
diamonds = pd.read_csv(url)

# Affichage des 5 premières lignes
print("Affichage des 5 premières lignes :")
print(diamonds.head())

# Affichage des noms de colonnes et types de données
print("\nInformation sur les colonnes et types de données :")
print(diamonds.info())

# Affichage des 10 premières lignes
print("\nAffichage des 10 premières lignes :")
print(diamonds.head(10))

# Création d'un récapitulatif des données pour une compréhension globale
print("\nRésumé statistique des données numériques :")
print(diamonds.describe())

# Comptage des diamants par catégorie de coupe
cut_counts = diamonds['cut'].value_counts().sort_index()
print("\nNombre de diamants par catégorie de coupe :")
print(cut_counts)

# Chargement des données
url = "https://raw.githubusercontent.com/TrainingByPackt/Interactive-Data-Visualization-with-Python/master/datasets/diamonds.csv"
diamonds = pd.read_csv(url)

#################################################
# 1. Matplotlib - Diagramme à barres
#################################################
plt.figure(figsize=(10, 6))
cut_counts = diamonds['cut'].value_counts().sort_index()
plt.bar(cut_counts.index, cut_counts.values, color='skyblue')
plt.xlabel('Qualité de la coupe')
plt.ylabel('Nombre de diamants')
plt.title('Nombre de diamants par qualité de coupe (Matplotlib)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
for i, v in enumerate(cut_counts.values):
    plt.text(i, v + 100, str(v), ha='center')
plt.tight_layout()
plt.show()

# Commentaire sur Matplotlib
print("""
Commentaire sur Matplotlib:
- Syntaxe impérative qui nécessite une configuration explicite de chaque élément
- Haute personnalisation possible mais code verbeux
- Contrôle total sur chaque aspect du graphique
- Nécessite des commandes supplémentaires pour des fonctionnalités avancées (annotations, rotation des étiquettes)
- Bon pour la production de visualisations statiques de qualité publication
""")

#################################################
# 2. Seaborn - Diagramme à barres
#################################################
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
ax = sns.countplot(x='cut', data=diamonds, palette='viridis')
ax.set_xlabel('Qualité de la coupe')
ax.set_ylabel('Nombre de diamants')
ax.set_title('Nombre de diamants par qualité de coupe (Seaborn)')
for i in ax.containers:
    ax.bar_label(i)
plt.tight_layout()
plt.show()

# Commentaire sur Seaborn
print("""
Commentaire sur Seaborn:
- Plus concis que Matplotlib avec des fonctions de haut niveau
- Style esthétique par défaut avec palette de couleurs attrayante
- Intégration directe avec pandas pour l'analyse statistique
- La fonction countplot() simplifie grandement la création de diagrammes de comptage
- Moins flexible que Matplotlib pour certaines personnalisations avancées
- Excellent pour l'exploration rapide de données et les visualisations statistiques
""")

#################################################
# 3. Plotly - Diagramme à barres interactif
#################################################
cut_counts_df = diamonds['cut'].value_counts().reset_index()
cut_counts_df.columns = ['cut', 'count']
cut_counts_df = cut_counts_df.sort_values('cut')

fig = px.bar(
    cut_counts_df, 
    x='cut', 
    y='count',
    text='count',
    title='Nombre de diamants par qualité de coupe (Plotly)',
    labels={'cut': 'Qualité de la coupe', 'count': 'Nombre de diamants'},
    color='cut',
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig.update_traces(textposition='outside')
fig.update_layout(
    xaxis_title='Qualité de la coupe',
    yaxis_title='Nombre de diamants',
    xaxis={'categoryorder': 'array', 'categoryarray': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']}
)
fig.show()

# Commentaire sur Plotly
print("""
Commentaire sur Plotly:
- Produit des visualisations interactives par défaut (zoom, survol, etc.)
- Syntaxe orientée objet avec méthode chaînée pour les configurations
- Excellent pour les dashboards et publications web
- Plus verbeux que Seaborn mais avec une interactivité supérieure
- Fonctionnalités avancées comme les info-bulles personnalisées et les animations
- Possibilité d'exporter en HTML pour partage facile
""")

#################################################
# 4. Altair - Diagramme à barres
#################################################
# Préparation des données pour Altair (compte les valeurs)
cut_counts_alt = diamonds['cut'].value_counts().reset_index()
cut_counts_alt.columns = ['cut', 'count']

# Création du diagramme avec Altair
chart = alt.Chart(cut_counts_alt).mark_bar().encode(
    x=alt.X('cut:N', sort=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], title='Qualité de la coupe'),
    y=alt.Y('count:Q', title='Nombre de diamants'),
    color=alt.Color('cut:N', legend=None),
    tooltip=['cut', 'count']
).properties(
    title='Nombre de diamants par qualité de coupe (Altair)',
    width=500,
    height=300
)

# Ajout des valeurs au-dessus des barres
text = chart.mark_text(
    align='center',
    baseline='bottom',
    dy=-5
).encode(
    text='count:Q'
)

# Combiner le graphique et le texte
final_chart = chart + text
final_chart

# Commentaire sur Altair
print("""
Commentaire sur Altair:
- Syntaxe déclarative basée sur la grammaire des graphiques
- Approche plus conceptuelle qui sépare les données et la représentation visuelle
- Expressivité élevée avec un code concis
- L'encodage des axes et des variables visuelles est très clair
- Interactivité facile à implémenter
- Bonne intégration avec les notebooks Jupyter et les applications web
""")

#################################################
# Analyse supplémentaire - Prix moyen par coupe
#################################################
# Calcul du prix moyen par type de coupe
price_by_cut = diamonds.groupby('cut')['price'].mean().reset_index()

# Matplotlib
plt.figure(figsize=(10, 6))
plt.bar(price_by_cut['cut'], price_by_cut['price'], color='salmon')
plt.xlabel('Qualité de la coupe')
plt.ylabel('Prix moyen ($)')
plt.title('Prix moyen des diamants par qualité de coupe')
plt.xticks(rotation=45)
for i, v in enumerate(price_by_cut['price']):
    plt.text(i, v + 50, f"${v:.2f}", ha='center')
plt.tight_layout()
plt.show()

# Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='cut', y='price', data=diamonds, estimator=np.mean, ci=None, palette='mako')
plt.xlabel('Qualité de la coupe')
plt.ylabel('Prix moyen ($)')
plt.title('Prix moyen des diamants par qualité de coupe (Seaborn)')
plt.tight_layout()
plt.show()

# Visualisation des diamants par clarté
clarity_counts = diamonds['clarity'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
sns.countplot(x='clarity', data=diamonds, palette='rocket')
plt.xlabel('Clarté')
plt.ylabel('Nombre de diamants')
plt.title('Distribution des diamants par niveau de clarté')
plt.tight_layout()
plt.show()

# Relation entre coupe, clarté et prix (heatmap avec Seaborn)
plt.figure(figsize=(12, 8))
cut_clarity_price = diamonds.groupby(['cut', 'clarity'])['price'].mean().unstack()
sns.heatmap(cut_clarity_price, annot=True, fmt='.0f', cmap='viridis')
plt.title('Prix moyen des diamants par coupe et clarté')
plt.tight_layout()
plt.show()