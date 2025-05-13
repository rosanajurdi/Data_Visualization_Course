import pandas as pd
import numpy as np
from datetime import datetime
from math import pi
from scipy import stats

from bokeh.plotting import figure, show
from bokeh.io import output_file, save, output_notebook
from bokeh.models import (ColumnDataSource, HoverTool, DatetimeTickFormatter, 
                         Band, Span, BoxAnnotation)
from bokeh.layouts import column, row, gridplot
from bokeh.palettes import Category10

# Pour une utilisation dans Jupyter Notebook
output_notebook()

# Chargement des données
def load_data():
    # Charger le fichier AirPassengersDates.csv
    # Assurez-vous que le fichier est dans le bon répertoire ou ajustez le chemin
    df = pd.read_csv('AirPassengersDates.csv')
    
    # Convertir la colonne de date en datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Définir la colonne Date comme index
    df.set_index('Date', inplace=True)
    
    return df

# Fonction pour créer un graphique de série temporelle de base
def create_time_series_plot(df):
    # Créer une source de données Bokeh
    source = ColumnDataSource(data={
        'date': df.index,
        'passengers': df['Passengers'],
        'date_str': [d.strftime('%Y-%m-%d') for d in df.index]
    })
    
    # Créer la figure
    p = figure(title='Nombre de passagers aériens (1949-1960)', 
               x_axis_type='datetime',
               x_axis_label='Date', 
               y_axis_label='Nombre de passagers',
               height=400, width=800)
    
    # Ajouter la ligne
    p.line('date', 'passengers', source=source, line_width=2, color='navy')
    
    # Ajouter les points
    p.circle('date', 'passengers', source=source, size=4, color='navy', alpha=0.5)
    
    # Ajouter un outil de survol
    hover = HoverTool(
        tooltips=[
            ('Date', '@date_str'),
            ('Passagers', '@passengers')
        ],
        mode='vline'
    )
    p.add_tools(hover)
    
    # Formatage de l'axe des dates
    p.xaxis.formatter = DatetimeTickFormatter(
        months=["%b %Y"],
        years=["%Y"]
    )
    p.xaxis.major_label_orientation = pi/4
    
    return p

# Fonction pour visualiser les agrégations (moyennes mensuelles/annuelles)
def create_aggregation_plots(df):
    # Agrégation mensuelle
    monthly_avg = df.groupby(df.index.month)['Passengers'].mean()
    monthly_data = pd.DataFrame({
        'month': range(1, 13),
        'average': monthly_avg.values,
        'month_name': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    })
    source_monthly = ColumnDataSource(monthly_data)
    
    # Graphique pour la moyenne mensuelle
    p1 = figure(title='Moyenne mensuelle de passagers', 
                x_range=monthly_data['month_name'].tolist(),
                y_axis_label='Nombre moyen de passagers',
                height=400, width=500)
    
    p1.vbar(x='month_name', top='average', source=source_monthly, width=0.8, 
            color='firebrick', alpha=0.7)
    
    hover1 = HoverTool(
        tooltips=[
            ('Mois', '@month_name'),
            ('Moyenne', '@average{0.0}')
        ]
    )
    p1.add_tools(hover1)
    
    # Agrégation annuelle
    yearly_avg = df.groupby(df.index.year)['Passengers'].mean()
    yearly_data = pd.DataFrame({
        'year': yearly_avg.index.astype(str),
        'average': yearly_avg.values
    })
    source_yearly = ColumnDataSource(yearly_data)
    
    # Graphique pour la moyenne annuelle
    p2 = figure(title='Moyenne annuelle de passagers', 
                x_range=yearly_data['year'].tolist(),
                y_axis_label='Nombre moyen de passagers',
                height=400, width=500)
    
    p2.vbar(x='year', top='average', source=source_yearly, width=0.8, 
            color='navy', alpha=0.7)
    
    hover2 = HoverTool(
        tooltips=[
            ('Année', '@year'),
            ('Moyenne', '@average{0.0}')
        ]
    )
    p2.add_tools(hover2)
    p2.xaxis.major_label_orientation = pi/4
    
    return row(p1, p2)

# Fonction pour visualiser la moyenne et l'écart-type
def create_mean_std_plot(df):
    # Calculer la moyenne mobile et l'écart-type
    window = 12  # 12 mois pour une moyenne mobile annuelle
    rolling_mean = df['Passengers'].rolling(window=window).mean()
    rolling_std = df['Passengers'].rolling(window=window).std()
    
    # Créer le DataFrame pour la visualisation
    roll_df = pd.DataFrame({
        'date': df.index,
        'passengers': df['Passengers'],
        'mean': rolling_mean,
        'upper': rolling_mean + 2*rolling_std,
        'lower': rolling_mean - 2*rolling_std,
        'date_str': [d.strftime('%Y-%m-%d') for d in df.index]
    }).dropna()
    
    source = ColumnDataSource(roll_df)
    
    # Créer la figure
    p = figure(title=f'Moyenne mobile ({window} mois) et écart-type', 
               x_axis_type='datetime',
               x_axis_label='Date', 
               y_axis_label='Nombre de passagers',
               height=400, width=800)
    
    # Ajouter la série originale
    p.line('date', 'passengers', source=source, line_width=1.5, 
           color='gray', legend_label='Données originales', alpha=0.7)
    
    # Ajouter la moyenne mobile
    p.line('date', 'mean', source=source, line_width=2.5, 
           color='navy', legend_label='Moyenne mobile')
    
    # Ajouter la bande pour l'écart-type
    band = Band(base='date', lower='lower', upper='upper', source=source,
                level='underlay', fill_alpha=0.3, fill_color='navy',
                line_width=1, line_color='black')
    p.add_layout(band)
    
    # Ajouter un outil de survol
    hover = HoverTool(
        tooltips=[
            ('Date', '@date_str'),
            ('Passagers', '@passengers'),
            ('Moyenne mobile', '@mean{0.0}'),
            ('Limite sup. (2σ)', '@upper{0.0}'),
            ('Limite inf. (2σ)', '@lower{0.0}')
        ],
        mode='vline'
    )
    p.add_tools(hover)
    
    # Formatage de l'axe des dates
    p.xaxis.formatter = DatetimeTickFormatter(
        months=["%b %Y"],
        years=["%Y"]
    )
    p.xaxis.major_label_orientation = pi/4
    
    # Configuration de la légende
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p

# Fonction pour la détection des valeurs aberrantes (outliers)
def create_outlier_plot(df):
    # Calculer Z-score pour identifier les outliers
    z_scores = stats.zscore(df['Passengers'])
    outliers = np.abs(z_scores) > 2.5  # Seuil de 2.5 pour les Z-scores
    
    # Créer un DataFrame avec des indicateurs d'outliers
    outlier_df = pd.DataFrame({
        'date': df.index,
        'passengers': df['Passengers'],
        'z_score': z_scores,
        'is_outlier': outliers,
        'date_str': [d.strftime('%Y-%m-%d') for d in df.index]
    })
    
    source = ColumnDataSource(outlier_df)
    
    # Créer la figure
    p = figure(title='Détection des valeurs aberrantes (Z-score > 2.5)', 
               x_axis_type='datetime',
               x_axis_label='Date', 
               y_axis_label='Nombre de passagers',
               height=400, width=800)
    
    # Tracer les points normaux
    p.circle('date', 'passengers', source=source, size=6, 
             color='navy', alpha=0.5, legend_label='Normal')
    
    # Tracer les outliers
    outliers = p.circle('date', 'passengers', source=ColumnDataSource(outlier_df[outlier_df['is_outlier']]), 
                       size=8, color='red', alpha=0.8, legend_label='Outlier')
    
    # Ajouter un outil de survol
    hover = HoverTool(
        tooltips=[
            ('Date', '@date_str'),
            ('Passagers', '@passengers'),
            ('Z-score', '@z_score{0.00}')
        ]
    )
    p.add_tools(hover)
    
    # Formatage de l'axe des dates
    p.xaxis.formatter = DatetimeTickFormatter(
        months=["%b %Y"],
        years=["%Y"]
    )
    p.xaxis.major_label_orientation = pi/4
    
    # Configuration de la légende
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p

# Fonction pour visualiser le resampling
def create_resampling_plots(df):
    # Downsampling (données trimestrielles)
    quarterly = df.resample('Q').mean()
    quarterly_df = pd.DataFrame({
        'date': quarterly.index,
        'passengers': quarterly['Passengers'],
        'date_str': [d.strftime('%Y-%m-%d') for d in quarterly.index]
    })
    source_quarterly = ColumnDataSource(quarterly_df)
    
    # Upsampling (données hebdomadaires par interpolation)
    weekly = df.resample('W').asfreq()
    weekly_interp = weekly.interpolate(method='linear')
    
    weekly_df = pd.DataFrame({
        'date': weekly_interp.index,
        'passengers': weekly_interp['Passengers'],
        'date_str': [d.strftime('%Y-%m-%d') for d in weekly_interp.index]
    })
    source_weekly = ColumnDataSource(weekly_df)
    
    # Graphique pour le downsampling
    p1 = figure(title='Downsampling (Trimestriel)', 
                x_axis_type='datetime',
                x_axis_label='Date', 
                y_axis_label='Nombre de passagers',
                height=400, width=800)
    
    # Données originales en arrière-plan
    p1.line(df.index, df['Passengers'], line_width=1, color='gray', 
            alpha=0.5, legend_label='Données mensuelles')
    
    # Données trimestrielles
    p1.circle('date', 'passengers', source=source_quarterly, size=8, 
              color='navy', alpha=0.8, legend_label='Moyenne trimestrielle')
    p1.line('date', 'passengers', source=source_quarterly, line_width=2, 
            color='navy', alpha=0.8)
    
    hover1 = HoverTool(
        tooltips=[
            ('Date', '@date_str'),
            ('Passagers (trim.)', '@passengers{0.0}')
        ]
    )
    p1.add_tools(hover1)
    
    # Graphique pour l'upsampling
    p2 = figure(title='Upsampling (Hebdomadaire avec interpolation)', 
                x_axis_type='datetime',
                x_axis_label='Date', 
                y_axis_label='Nombre de passagers',
                height=400, width=800)
    
    # Données originales
    p2.circle(df.index, df['Passengers'], size=6, color='red', 
              alpha=0.8, legend_label='Données mensuelles')
    
    # Données hebdomadaires interpolées
    p2.line('date', 'passengers', source=source_weekly, line_width=1.5, 
            color='navy', alpha=0.6, legend_label='Interpolation hebdomadaire')
    
    hover2 = HoverTool(
        tooltips=[
            ('Date', '@date_str'),
            ('Passagers (hebdo.)', '@passengers{0.0}')
        ]
    )
    p2.add_tools(hover2)
    
    # Formatage des axes des dates
    for p in [p1, p2]:
        p.xaxis.formatter = DatetimeTickFormatter(
            months=["%b %Y"],
            years=["%Y"]
        )
        p.xaxis.major_label_orientation = pi/4
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
    
    return column(p1, p2)

# Fonction pour l'analyse des lags (retards)
def create_lag_plot(df):
    # Créer des lags pour 1, 6 et 12 mois
    lag_periods = [1, 6, 12]
    plots = []
    
    for lag in lag_periods:
        # Créer les données avec lag
        lag_data = pd.DataFrame({
            'original': df['Passengers'].values[lag:],
            'lagged': df['Passengers'].values[:-lag],
        })
        lag_data['month'] = list(range(lag, len(df)))
        lag_data['date'] = df.index[lag:]
        lag_data['date_str'] = [d.strftime('%Y-%m-%d') for d in lag_data['date']]
        
        source = ColumnDataSource(lag_data)
        
        # Calculer la corrélation
        corr = np.corrcoef(lag_data['original'], lag_data['lagged'])[0, 1]
        
        # Créer le scatter plot
        p = figure(title=f'Lag Plot (Lag = {lag} mois, Corrélation = {corr:.3f})', 
                   x_axis_label=f'Passagers (t-{lag})', 
                   y_axis_label='Passagers (t)',
                   height=300, width=300)
        
        p.circle('lagged', 'original', source=source, size=6, 
                color='navy', alpha=0.6)
        
        # Ajouter une ligne de régression simple
        m, b = np.polyfit(lag_data['lagged'], lag_data['original'], 1)
        x_range = np.linspace(min(lag_data['lagged']), max(lag_data['lagged']), 100)
        y_range = m * x_range + b
        p.line(x_range, y_range, line_width=2, color='red', 
               legend_label=f'y = {m:.2f}x + {b:.0f}')
        
        # Ajouter une ligne de référence y=x
        p.line([min(lag_data['lagged']), max(lag_data['lagged'])], 
               [min(lag_data['lagged']), max(lag_data['lagged'])], 
               line_width=1, color='gray', line_dash='dashed', alpha=0.7)
        
        # Ajouter un outil de survol
        hover = HoverTool(
            tooltips=[
                ('Date', '@date_str'),
                (f'Passagers (t-{lag})', '@lagged'),
                ('Passagers (t)', '@original')
            ]
        )
        p.add_tools(hover)
        
        # Configuration de la légende
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        
        plots.append(p)
    
    # Disposer les plots dans une grille
    grid = gridplot([plots], ncols=3)
    
    return grid

# Fonction pour visualiser l'autocorrélation
def create_autocorrelation_plot(df):
    # Fonction manuelle pour calculer l'autocorrélation
    def manual_acf(series, nlags):
        result = np.zeros(nlags + 1)
        y = series - series.mean()
        y_var = np.var(y)
        n = len(y)
        
        for lag in range(nlags + 1):
            cross_sum = 0
            for i in range(n - lag):
                cross_sum += y[i] * y[i + lag]
            
            result[lag] = cross_sum / ((n - lag) * y_var)
        
        return result
    
    # Calculer l'autocorrélation
    n_lags = 36  # 3 ans
    acf_values = manual_acf(df['Passengers'].values, n_lags)
    confidence = 1.96 / np.sqrt(len(df))  # Intervalle de confiance de 95%
    
    # Créer les données pour le graphique
    acf_data = pd.DataFrame({
        'lag': range(len(acf_values)),
        'acf': acf_values,
        'upper_ci': confidence,
        'lower_ci': -confidence
    })
    
    source = ColumnDataSource(acf_data)
    
    # Créer la figure
    p = figure(title=f'Fonction d\'autocorrélation (ACF)', 
               x_axis_label='Lag (mois)', 
               y_axis_label='Autocorrélation',
               height=400, width=800)
    
    # Ajouter les barres d'autocorrélation
    p.vbar(x='lag', top='acf', source=source, width=0.7, 
           color='navy', alpha=0.7)
    
    # Ajouter les lignes d'intervalle de confiance
    upper_ci = Span(location=confidence, dimension='width', 
                    line_color='red', line_dash='dashed', line_width=2)
    lower_ci = Span(location=-confidence, dimension='width', 
                   line_color='red', line_dash='dashed', line_width=2)
    
    p.add_layout(upper_ci)
    p.add_layout(lower_ci)
    
    # Ajouter une ligne à zéro
    zero_line = Span(location=0, dimension='width', 
                    line_color='black', line_width=1)
    p.add_layout(zero_line)
    
    # Ajouter un outil de survol
    hover = HoverTool(
        tooltips=[
            ('Lag', '@lag mois'),
            ('ACF', '@acf{0.000}')
        ]
    )
    p.add_tools(hover)
    
    return p

# Fonction principale pour exécuter l'analyse complète
def run_time_series_analysis():
    # Charger les données
    df = load_data()
    
    # Créer tous les graphiques
    ts_plot = create_time_series_plot(df)
    agg_plots = create_aggregation_plots(df)
    mean_std_plot = create_mean_std_plot(df)
    outlier_plot = create_outlier_plot(df)
    resampling_plots = create_resampling_plots(df)
    lag_plot = create_lag_plot(df)
    acf_plot = create_autocorrelation_plot(df)
    
    # Afficher tous les graphiques
    # Pour un notebook:
    show(column(
        ts_plot, 
        agg_plots, 
        mean_std_plot, 
        outlier_plot, 
        resampling_plots,
        lag_plot,
        acf_plot
    ))
    
    # Pour sauvegarder en HTML:
    output_file("air_passengers_bokeh_analysis.html")
    save(column(
        ts_plot, 
        agg_plots, 
        mean_std_plot, 
        outlier_plot, 
        resampling_plots,
        lag_plot,
        acf_plot
    ))

if __name__ == "__main__":
    run_time_series_analysis()