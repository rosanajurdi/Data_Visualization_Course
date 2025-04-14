



# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Chargement du jeu de données diamonds
# Note: Le jeu de données diamonds est disponible dans seaborn
diamonds = sns.load_dataset('diamonds')

# Affichons les premières lignes pour comprendre la structure des données
print("Aperçu du jeu de données diamonds:")
print(diamonds.head())
print("\nInformations sur le jeu de données:")
print(diamonds.info())
print("\nStatistiques descriptives de la colonne 'carat':")
print(diamonds['carat'].describe())

# 1. Création d'un histogramme simple avec pandas
plt.figure(figsize=(10, 6))
diamonds['carat'].hist()
plt.title('Histogramme du poids des diamants (carats)')
plt.xlabel('Poids (carats)')
plt.ylabel('Fréquence')
plt.grid(False)
plt.show()

# 2. Modification du nombre de bins
plt.figure(figsize=(15, 10))

# Histogramme avec 10 bins
plt.subplot(2, 2, 1)
diamonds['carat'].hist(bins=10, color='skyblue')
plt.title('Histogramme avec 10 bins')
plt.xlabel('Poids (carats)')
plt.ylabel('Fréquence')
plt.grid(False)

# Histogramme avec 30 bins
plt.subplot(2, 2, 2)
diamonds['carat'].hist(bins=30, color='green')
plt.title('Histogramme avec 30 bins')
plt.xlabel('Poids (carats)')
plt.ylabel('Fréquence')
plt.grid(False)

# Histogramme avec 50 bins
plt.subplot(2, 2, 3)
diamonds['carat'].hist(bins=50, color='orange')
plt.title('Histogramme avec 50 bins')
plt.xlabel('Poids (carats)')
plt.ylabel('Fréquence')
plt.grid(False)

# Histogramme avec 100 bins
plt.subplot(2, 2, 4)
diamonds['carat'].hist(bins=100, color='red')
plt.title('Histogramme avec 100 bins')
plt.xlabel('Poids (carats)')
plt.ylabel('Fréquence')
plt.grid(False)

plt.tight_layout()
plt.show()

# 4. Tracer un histogramme avec Seaborn
plt.figure(figsize=(10, 6))
sns.histplot(data=diamonds, x='carat', kde=True)
plt.title('Histogramme du poids des diamants avec KDE (Seaborn)')
plt.xlabel('Poids (carats)')
plt.ylabel('Fréquence')
plt.grid(False)
plt.show()

# Comparaison entre histogramme pandas et seaborn
plt.figure(figsize=(15, 6))

# Histogramme pandas
plt.subplot(1, 2, 1)
diamonds['carat'].hist(bins=30, color='skyblue')
plt.title('Histogramme avec Pandas')
plt.xlabel('Poids (carats)')
plt.ylabel('Fréquence')
plt.grid(False)

# Histogramme seaborn
plt.subplot(1, 2, 2)
sns.histplot(data=diamonds, x='carat', bins=30, kde=True, color='skyblue')
plt.title('Histogramme avec Seaborn (avec KDE)')
plt.xlabel('Poids (carats)')
plt.ylabel('Fréquence')
plt.grid(False)

plt.tight_layout()
plt.show()

# 6. Visualisation du KDE seul
plt.figure(figsize=(10, 6))
sns.kdeplot(data=diamonds, x='carat', fill=True)
plt.title('Estimation de la densité du noyau (KDE) du poids des diamants')
plt.xlabel('Poids (carats)')
plt.ylabel('Densité')
plt.grid(False)
plt.show()

# 7. Application d'une transformation logarithmique
plt.figure(figsize=(15, 6))

# Histogramme standard
plt.subplot(1, 2, 1)
sns.histplot(data=diamonds, x='carat', kde=True)
plt.title('Histogramme standard')
plt.xlabel('Poids (carats)')
plt.ylabel('Fréquence')
plt.grid(False)

# Histogramme avec transformation logarithmique
plt.subplot(1, 2, 2)
# Ajout d'une petite valeur pour éviter log(0)
sns.histplot(data=diamonds, x='carat', kde=True, log_scale=True)
plt.title('Histogramme avec échelle logarithmique')
plt.xlabel('Poids (carats) - échelle logarithmique')
plt.ylabel('Fréquence')
plt.grid(False)

plt.tight_layout()
plt.show()

# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Chargement du jeu de données diamonds
diamonds = sns.load_dataset('diamonds')

# Affichons les premières lignes et statistiques sur les prix
print("Aperçu du jeu de données diamonds:")
print(diamonds.head())
print("\nStatistiques descriptives de la colonne 'price':")
print(diamonds['price'].describe())

# 1. Histogramme simple des prix
plt.figure(figsize=(10, 6))
diamonds['price'].hist(bins=30)
plt.title('Histogramme des prix des diamants')
plt.xlabel('Prix ($)')
plt.ylabel('Fréquence')
plt.grid(False)
plt.show()

# 2. Histogramme des prix avec différents nombres de bins
plt.figure(figsize=(15, 10))

# Histogramme avec 10 bins
plt.subplot(2, 2, 1)
diamonds['price'].hist(bins=10, color='skyblue')
plt.title('Prix des diamants - 10 bins')
plt.xlabel('Prix ($)')
plt.ylabel('Fréquence')
plt.grid(False)

# Histogramme avec 30 bins
plt.subplot(2, 2, 2)
diamonds['price'].hist(bins=30, color='green')
plt.title('Prix des diamants - 30 bins')
plt.xlabel('Prix ($)')
plt.ylabel('Fréquence')
plt.grid(False)

# Histogramme avec 50 bins
plt.subplot(2, 2, 3)
diamonds['price'].hist(bins=50, color='orange')
plt.title('Prix des diamants - 50 bins')
plt.xlabel('Prix ($)')
plt.ylabel('Fréquence')
plt.grid(False)

# Histogramme avec 100 bins
plt.subplot(2, 2, 4)
diamonds['price'].hist(bins=100, color='red')
plt.title('Prix des diamants - 100 bins')
plt.xlabel('Prix ($)')
plt.ylabel('Fréquence')
plt.grid(False)

plt.tight_layout()
plt.show()

# 3. Comparaison entre distribution normale et logarithmique des prix
plt.figure(figsize=(15, 6))

# Histogramme standard
plt.subplot(1, 2, 1)
sns.histplot(data=diamonds, x='price', kde=True, bins=30, color='skyblue')
plt.title('Distribution originale des prix')
plt.xlabel('Prix ($)')
plt.ylabel('Fréquence')
plt.grid(False)

# Histogramme avec transformation logarithmique
plt.subplot(1, 2, 2)
sns.histplot(data=diamonds, x='price', kde=True, bins=30, color='green', log_scale=True)
plt.title('Distribution logarithmique des prix')
plt.xlabel('Prix ($) - échelle logarithmique')
plt.ylabel('Fréquence')
plt.grid(False)

plt.tight_layout()
plt.show()

# 4. Création d'une colonne avec le log des prix pour une analyse plus détaillée
diamonds['log_price'] = np.log(diamonds['price'])

# Histogramme de log_price pour mieux visualiser les pics
plt.figure(figsize=(12, 6))
sns.histplot(data=diamonds, x='log_price', kde=True, bins=50)
plt.title('Distribution du logarithme des prix des diamants')
plt.xlabel('Log(Prix)')
plt.ylabel('Fréquence')
plt.grid(False)

# Ajoutons des lignes verticales pour identifier les pics approximatifs
plt.axvline(x=6.8, color='red', linestyle='--', alpha=0.7, label='Pic 1 (~6.8)')
plt.axvline(x=8.5, color='blue', linestyle='--', alpha=0.7, label='Pic 2 (~8.5)')
plt.axvline(x=9.0, color='green', linestyle='--', alpha=0.7, label='Pic 3 (~9.0)')
plt.legend()
plt.show()

# 5. Conversion des valeurs logarithmiques en prix réels pour meilleure interprétation
log_values = [6.8, 8.5, 9.0]
real_prices = [np.exp(val) for val in log_values]

print("\nPics identifiés sur l'échelle logarithmique et leurs prix correspondants:")
for log_val, real_price in zip(log_values, real_prices):
    print(f"Log(Prix) = {log_val} correspond à Prix = ${real_price:.2f}")

# 6. Analyse des diamants autour des pics identifiés
price_ranges = [
    (np.exp(6.7), np.exp(6.9)),  # Autour du premier pic
    (np.exp(8.4), np.exp(8.6)),  # Autour du deuxième pic
    (np.exp(8.9), np.exp(9.1))   # Autour du troisième pic
]

# Affichons les caractéristiques des diamants autour de chaque pic
for i, (lower, upper) in enumerate(price_ranges):
    print(f"\nCaractéristiques des diamants autour du pic {i+1} (prix entre ${lower:.2f} et ${upper:.2f}):")
    subset = diamonds[(diamonds['price'] >= lower) & (diamonds['price'] <= upper)]
    print(subset[['carat', 'cut', 'color', 'clarity', 'price']].describe().round(2))
    
# 7. Visualisation de la relation entre prix et qualité
plt.figure(figsize=(14, 8))

# Prix vs. carat avec coloration par qualité de coupe
plt.subplot(2, 2, 1)
sns.scatterplot(data=diamonds, x='carat', y='price', hue='cut', alpha=0.5)
plt.title('Prix vs. Carat (coloré par qualité de coupe)')
plt.xlabel('Carat')
plt.ylabel('Prix ($)')

# Prix vs. carat avec coloration par couleur
plt.subplot(2, 2, 2)
sns.scatterplot(data=diamonds, x='carat', y='price', hue='color', alpha=0.5)
plt.title('Prix vs. Carat (coloré par couleur)')
plt.xlabel('Carat')
plt.ylabel('Prix ($)')

# Prix vs. carat avec coloration par clarté
plt.subplot(2, 2, 3)
sns.scatterplot(data=diamonds, x='carat', y='price', hue='clarity', alpha=0.5)
plt.title('Prix vs. Carat (coloré par clarté)')
plt.xlabel('Carat')
plt.ylabel('Prix ($)')

# Boxplot des prix par qualité de coupe (échelle logarithmique)
plt.subplot(2, 2, 4)
sns.boxplot(data=diamonds, x='cut', y='price', log_scale=True)
plt.title('Distribution des prix par qualité de coupe (échelle log)')
plt.xlabel('Qualité de coupe')
plt.ylabel('Prix ($) - échelle logarithmique')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Importation des bibliothèques nécessaires
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# Configuration pour afficher les graphiques dans le navigateur si nécessaire
import plotly.io as pio
pio.renderers.default = "browser"  # Commentez cette ligne si vous utilisez un notebook Jupyter

# Chargement du jeu de données diamonds
diamonds = sns.load_dataset('diamonds')

# Affichons les premières lignes et statistiques sur les données
print("Aperçu du jeu de données diamonds:")
print(diamonds.head())
print("\nStatistiques descriptives de la colonne 'carat':")
print(diamonds['carat'].describe())
print("\nStatistiques descriptives de la colonne 'price':")
print(diamonds['price'].describe())

# 1. Histogramme simple des carats avec Plotly - définir mais ne pas afficher immédiatement
fig_carat = px.histogram(diamonds, x="carat", 
                   title='Histogramme du poids des diamants (carats)',
                   labels={'carat': 'Poids (carats)', 'count': 'Fréquence'},
                   opacity=0.7)
fig_carat.update_layout(
    xaxis_title="Poids (carats)",
    yaxis_title="Fréquence",
    template="plotly_white"
)
# Enregistrer la figure au lieu de l'afficher
fig_carat.write_html("histogramme_carat.html")
print("Histogramme du poids des diamants généré dans 'histogramme_carat.html'")

# 2. Modification du nombre de bins pour l'histogramme des carats
fig_bins = make_subplots(rows=2, cols=2, 
                    subplot_titles=['10 bins', '30 bins', '50 bins', '100 bins'])

# Histogramme avec 10 bins
fig_bins.add_trace(
    go.Histogram(x=diamonds['carat'], nbinsx=10, marker_color='skyblue', 
                 name='10 bins', opacity=0.7),
    row=1, col=1
)

# Histogramme avec 30 bins
fig_bins.add_trace(
    go.Histogram(x=diamonds['carat'], nbinsx=30, marker_color='green', 
                 name='30 bins', opacity=0.7),
    row=1, col=2
)

# Histogramme avec 50 bins
fig_bins.add_trace(
    go.Histogram(x=diamonds['carat'], nbinsx=50, marker_color='orange', 
                 name='50 bins', opacity=0.7),
    row=2, col=1
)

# Histogramme avec 100 bins
fig_bins.add_trace(
    go.Histogram(x=diamonds['carat'], nbinsx=100, marker_color='red', 
                 name='100 bins', opacity=0.7),
    row=2, col=2
)

fig_bins.update_layout(
    title_text="Histogrammes du poids des diamants avec différents nombres de bins",
    showlegend=False,
    height=700,
    template="plotly_white"
)

# Mise à jour des axes
for i in range(1, 3):
    for j in range(1, 3):
        fig_bins.update_xaxes(title_text="Poids (carats)", row=i, col=j)
        fig_bins.update_yaxes(title_text="Fréquence", row=i, col=j)

# Enregistrer au lieu d'afficher
fig_bins.write_html("histogramme_bins_comparison.html")
print("Comparaison des bins générée dans 'histogramme_bins_comparison.html'")

# Création des graphiques pour l'analyse des prix
# Histogramme des prix original et logarithmique
fig_prices = make_subplots(rows=1, cols=2, 
                    subplot_titles=['Distribution originale des prix', 'Distribution logarithmique des prix'])

# Histogramme standard des prix
fig_prices.add_trace(
    go.Histogram(x=diamonds['price'], nbinsx=30, marker_color='orange', opacity=0.7),
    row=1, col=1
)

# Histogramme des prix avec échelle logarithmique
fig_prices.add_trace(
    go.Histogram(x=diamonds['price'], nbinsx=30, marker_color='green', opacity=0.7),
    row=1, col=2
)

fig_prices.update_layout(
    title_text="Histogrammes des prix des diamants: normal vs. logarithmique",
    showlegend=False,
    height=500,
    template="plotly_white"
)

fig_prices.update_xaxes(title_text="Prix ($)", row=1, col=1)
fig_prices.update_yaxes(title_text="Fréquence", row=1, col=1)
fig_prices.update_xaxes(title_text="Prix ($)", type="log", row=1, col=2)
fig_prices.update_yaxes(title_text="Fréquence", row=1, col=2)

# Enregistrer au lieu d'afficher
fig_prices.write_html("prix_comparaison.html")
print("Comparaison des prix générée dans 'prix_comparaison.html'")

# Création d'une colonne avec le log des prix pour une analyse plus détaillée
diamonds['log_price'] = np.log(diamonds['price'])

# Histogramme de log_price avec identification des pics
fig_log_prices = px.histogram(diamonds, x="log_price", nbins=50,
                  title="Distribution du logarithme des prix des diamants",
                  labels={'log_price': 'Log(Prix)', 'count': 'Fréquence'},
                  opacity=0.7)

# Ajout de lignes verticales pour marquer les pics
fig_log_prices.add_vline(x=6.8, line_width=2, line_dash="dash", line_color="red")
fig_log_prices.add_vline(x=8.5, line_width=2, line_dash="dash", line_color="blue")
fig_log_prices.add_vline(x=9.0, line_width=2, line_dash="dash", line_color="green")

# Ajouter des annotations pour les pics
fig_log_prices.add_annotation(x=6.8, y=300, text="Pic 1 (~$900)", showarrow=True, arrowhead=1, ax=50, ay=-30)
fig_log_prices.add_annotation(x=8.5, y=300, text="Pic 2 (~$4,900)", showarrow=True, arrowhead=1, ax=-50, ay=-50)
fig_log_prices.add_annotation(x=9.0, y=300, text="Pic 3 (~$8,100)", showarrow=True, arrowhead=1, ax=-50, ay=-70)

fig_log_prices.update_layout(
    xaxis_title="Log(Prix)",
    yaxis_title="Fréquence",
    template="plotly_white"
)

# Enregistrer au lieu d'afficher
fig_log_prices.write_html("log_prix_pics.html")
print("Histogramme du logarithme des prix généré dans 'log_prix_pics.html'")

# Calcul des prix réels correspondant aux pics logarithmiques
log_values = [6.8, 8.5, 9.0]
real_prices = [np.exp(val) for val in log_values]

print("\nPics identifiés sur l'échelle logarithmique et leurs prix correspondants:")
for log_val, real_price in zip(log_values, real_prices):
    print(f"Log(Prix) = {log_val} correspond à Prix = ${real_price:.2f}")

# Enregistrement d'un graphique interactif pour l'analyse des relations
fig_scatter = px.scatter(diamonds, x="carat", y="price", color="clarity", 
                 hover_data=["cut", "color", "depth", "table"],
                 title="Prix vs. Carat (coloré par clarté)",
                 labels={"carat": "Poids (carats)", "price": "Prix ($)", 
                        "clarity": "Clarté", "cut": "Coupe", "color": "Couleur"},
                 opacity=0.6)

# Option pour exporter un graphique avec contrôles interactifs
fig_scatter.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.1,
            y=1.15,
            buttons=list([
                dict(label="Échelle normale",
                     method="relayout",
                     args=[{"xaxis.type": "linear", "yaxis.type": "linear"}]),
                dict(label="Y log",
                     method="relayout",
                     args=[{"xaxis.type": "linear", "yaxis.type": "log"}]),
            ]),
        )
    ],
    template="plotly_white"
)

fig_scatter.write_html("prix_vs_carat.html")
print("Graphique interactif Prix vs Carat généré dans 'prix_vs_carat.html'")

print("\nTous les graphiques ont été générés avec succès sous forme de fichiers HTML.")
print("Vous pouvez ouvrir ces fichiers dans votre navigateur pour visualiser les graphiques interactifs.")