import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# Activate notebook output
output_notebook()

# Load data
df = pd.read_csv("C:/Users/victo/Documents/GitHub/Data_Visualization_Course/Data_Visualization_Course/lab-sessions/datasets/AirPassengersDates.csv")
df.rename(columns={'#Passengers': 'Passengers'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df.set_index('Date', inplace=True)

# ----------------------
# 1. Annual bar plot
# ----------------------
df_yearly = df.resample('YE').sum()
df_yearly.index = df_yearly.index.year.astype(str)
yearly_source = ColumnDataSource(df_yearly.reset_index())

bar_plot = figure(x_range=yearly_source.data['Date'],
                 title="Total Passengers per Year",
                 x_axis_label='Year', y_axis_label='Total Passengers',
                 height=300, width=800, tools="pan,box_zoom,reset,save")

bar_plot.vbar(x='Date', top='Passengers', width=0.8, source=yearly_source, color="orange")
bar_plot.add_tools(HoverTool(tooltips=[("Year", "@Date"), ("Passengers", "@Passengers")]))

# ----------------------
# 2. Mean + std dev + outliers
# ----------------------
mean = df['Passengers'].mean()
std = df['Passengers'].std()

mean_plot = figure(title="Passengers with Mean and ±1 Std Dev",
                 x_axis_type='datetime', x_axis_label='Date', y_axis_label='Passengers',
                 height=300, width=800, tools="pan,box_zoom,reset,save")

mean_plot.line(df.index, df['Passengers'], line_width=2, color="green", legend_label="Passengers")
mean_plot.line(df.index, [mean]*len(df), line_dash='dashed', color='black', legend_label="Mean")
mean_plot.line(df.index, [mean + std]*len(df), line_dash='dotted', color='gray', legend_label="+1 Std Dev")
mean_plot.line(df.index, [mean - std]*len(df), line_dash='dotted', color='gray', legend_label="-1 Std Dev")

outliers = df[(df['Passengers'] > mean + 2*std) | (df['Passengers'] < mean - 2*std)]
outlier_source = ColumnDataSource(data={
    'date': outliers.index,
    'passengers': outliers['Passengers']
})
# Updated from circle() to scatter()
mean_plot.scatter('date', 'passengers', size=8, source=outlier_source, color='red', legend_label="Outliers")

# ----------------------
# 3. Resampling (down & up)
# ----------------------
df_downsampled = df.resample('YE').mean()
df_upsampled = df.resample('W').interpolate('linear')

resample_plot = figure(title="Resampling: Yearly Mean & Weekly Interpolation",
                      x_axis_type='datetime', height=300, width=800)

resample_plot.line(df.index, df['Passengers'], color="lightgray", legend_label="Original")
resample_plot.line(df_downsampled.index, df_downsampled['Passengers'], color="blue", line_width=2, legend_label="Yearly Mean")
resample_plot.line(df_upsampled.index, df_upsampled['Passengers'], color="green", line_width=1, alpha=0.6, legend_label="Weekly Interpolation")

# ----------------------
# 4. Autocorrelation (matplotlib)
# ----------------------
fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(df['Passengers'], ax=ax, lags=40)
plt.title("Autocorrelation of Air Passengers")
plt.tight_layout()

# ----------------------
# 5. Display everything
# ----------------------
# Show Bokeh plots
show(column(bar_plot, mean_plot, resample_plot))

# Show matplotlib plot
plt.show()

print("📊 All visualizations displayed in notebook")