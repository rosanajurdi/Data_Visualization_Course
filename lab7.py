import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    DatetimeTickFormatter,
    NumeralTickFormatter,
    RangeTool,
)
from bokeh.layouts import row, column
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral11
import numpy as np
from statistics import median

# Enable Bokeh output in Jupyter Notebook
output_notebook()

# --- Data Loading and Preprocessing ---
print("--- Loading and Preprocessing Data ---")
# Load the Dataset - Adjust the path as needed
try:
    df = pd.read_csv("daily-minimum-temperatures-in-melbourne.csv")
except FileNotFoundError:
    try:
        df = pd.read_csv("./datasets/daily-minimum-temperatures-in-melbourne.csv")
    except FileNotFoundError:
        # Fallback for specific environment (update this if your path is different)
        # Please adjust this path if neither of the above works for you.
        # This path is an example, you might need to change it.
        df = pd.read_csv("C:/Users/victo/Documents/GitHub/Data_Visualization_Course/Data_Visualization_Course/datasets/daily-minimum-temperatures-in-melbourne.csv")

# Rename columns for clarity
df.columns = ['Date', 'Temperature']

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Remove '?' from the 'Temperature' column and convert to numeric
df['Temperature'] = df['Temperature'].astype(str).str.replace('?', '', regex=False)
df['Temperature'] = pd.to_numeric(df['Temperature'])

print("Data loaded and preprocessed successfully. Displaying first 5 rows:")
print(df.head())
print("\n" + "="*80 + "\n")

# ---
# # Exam: Time Series Visualization with Bokeh
#
# This exam tests your ability to visualize time series data using the Bokeh library. You'll be working with the "Daily Minimum Temperatures in Melbourne" dataset. For each question, provide the Python code using Bokeh to generate the requested visualization.
#
# **Dataset:** "daily-minimum-temperatures-in-melbourne.csv"
#
# The initial setup code for loading and preprocessing the dataset is included at the beginning of this script. Each question's solution is embedded directly below its description.

# ---
# ### Question 1: Basic Time Series Line Plot
#
# 1.  Create a basic line plot showing the daily minimum temperature over time.
#
#     * Use the **'Date'** column on the x-axis and the **'Temperature'** column on the y-axis.
#     * Set the plot title to **"Daily Minimum Temperatures"**.
#     * Label the x-axis as **"Date"** and the y-axis as **"Temperature (°C)"**.
#     * Add **tooltips** to display the date and temperature when hovering over the line.
#     * Enable **pan, wheel zoom, and reset tools**.

print("--- Displaying Question 1: Basic Time Series Line Plot ---")
source_q1 = ColumnDataSource(df)

p_q1 = figure(
    title="Daily Minimum Temperatures",
    x_axis_label="Date",
    y_axis_label="Temperature (°C)",
    x_axis_type="datetime",
    height=400,
    width=800,
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

p_q1.line(
    x='Date',
    y='Temperature',
    source=source_q1,
    line_width=2,
    color='navy'
)

hover_q1 = HoverTool(
    tooltips=[
        ("Date", "@Date{%F}"),
        ("Temperature", "@Temperature{0.0} °C")
    ],
    formatters={
        "@Date": "datetime"
    },
    mode="vline"
)
p_q1.add_tools(hover_q1)

p_q1.xaxis.formatter = DatetimeTickFormatter(
    days="%d %b %Y",
    months="%b %Y",
    years="%Y"
)
p_q1.yaxis.formatter = NumeralTickFormatter(format="0.0")

show(p_q1)
print("\n" + "="*80 + "\n")

# ---
# ### Question 2: Rolling Average
#
# 2.  Calculate the 30-day rolling average of the daily minimum temperature and plot it
#     alongside the original temperature data.
#
#     * Create a new column **'Rolling_Avg'** in the DataFrame containing the 30-day rolling average.
#     * Plot both the original **'Temperature'** and the **'Rolling_Avg'** on the same plot.
#     * Use different colors and line styles to distinguish between the two.
#     * Add a **legend** to the plot to label the lines.
#     * Add **tooltips** to display the date, original temperature, and rolling average.

print("--- Displaying Question 2: Rolling Average Plot ---")
df['Rolling_Avg'] = df['Temperature'].rolling(window=30).mean()

source_q2 = ColumnDataSource(df)

p_q2 = figure(
    title="Daily Minimum Temperatures with 30-Day Rolling Average",
    x_axis_label="Date",
    y_axis_label="Temperature (°C)",
    x_axis_type="datetime",
    height=400,
    width=800,
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

p_q2.line(
    x='Date',
    y='Temperature',
    source=source_q2,
    line_width=1,
    color='navy',
    alpha=0.7,
    legend_label="Daily Temperature"
)

p_q2.line(
    x='Date',
    y='Rolling_Avg',
    source=source_q2,
    line_width=2,
    color='red',
    line_dash='solid',
    legend_label="30-Day Rolling Average"
)

hover_q2 = HoverTool(
    tooltips=[
        ("Date", "@Date{%F}"),
        ("Temperature", "@Temperature{0.0} °C"),
        ("Rolling Avg", "@Rolling_Avg{0.00} °C")
    ],
    formatters={
        "@Date": "datetime"
    },
    mode="vline"
)
p_q2.add_tools(hover_q2)

p_q2.xaxis.formatter = DatetimeTickFormatter(
    days="%d %b %Y",
    months="%b %Y",
    years="%Y"
)
p_q2.yaxis.formatter = NumeralTickFormatter(format="0.0")

p_q2.legend.location = "top_left"
p_q2.legend.click_policy = "hide"

show(p_q2)
print("\n" + "="*80 + "\n")

# ---
# ### Question 3: Monthly Box Plots
#
# 3.  Create box plots to visualize the distribution of temperatures for each month.
#
#     * Extract the **month** from the 'Date' column and create a new **'Month'** column.
#     * Group the data by **'Month'** and prepare it for plotting.
#     * Use Bokeh's box plot elements to visualize the distribution.
#     * Label the x-axis with month names and the y-axis with **"Temperature (°C)"**.
#     * Add **tooltips** to display the month and relevant statistical values (min, max, median, etc.).

print("--- Displaying Question 3: Monthly Box Plots ---")
df['Month'] = df['Date'].dt.month_name()

month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

groups_q3 = df.groupby('Month')

cats_q3 = []
q1s_q3 = []
q2s_q3 = []
q3s_q3 = []
uppers_q3 = []
lowers_q3 = []

for month_name in month_order:
    group = groups_q3.get_group(month_name)
    q1 = group['Temperature'].quantile(0.25)
    q2 = group['Temperature'].quantile(0.5)
    q3 = group['Temperature'].quantile(0.75)
    iqr = q3 - q1
    upper = min(group['Temperature'].max(), q3 + 1.5 * iqr)
    lower = max(group['Temperature'].min(), q1 - 1.5 * iqr)

    cats_q3.append(month_name)
    q1s_q3.append(q1)
    q2s_q3.append(q2)
    q3s_q3.append(q3)
    uppers_q3.append(upper)
    lowers_q3.append(lower)

source_q3 = ColumnDataSource(data=dict(
    cats=cats_q3,
    q1=q1s_q3,
    q2=q2s_q3,
    q3=q3s_q3,
    upper=uppers_q3,
    lower=lowers_q3
))

outliers_q3 = []
outlier_months_q3 = []

for month_name in month_order:
    group = groups_q3.get_group(month_name)
    q1 = group['Temperature'].quantile(0.25)
    q3 = group['Temperature'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr

    outlier_idx = (group['Temperature'] > upper_bound) | (group['Temperature'] < lower_bound)
    month_outliers = group.loc[outlier_idx, 'Temperature'].tolist()
    outliers_q3.extend(month_outliers)
    outlier_months_q3.extend([month_name] * len(month_outliers))

outlier_source_q3 = ColumnDataSource(data=dict(
    month=outlier_months_q3,
    temp=outliers_q3
))

p_q3 = figure(
    title="Monthly Distribution of Minimum Temperatures",
    x_range=month_order,
    x_axis_label="Month",
    y_axis_label="Temperature (°C)",
    height=500,
    width=800,
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

# Stems
p_q3.segment('cats', 'upper', 'cats', 'q3', source=source_q3, line_color="black")
p_q3.segment('cats', 'lower', 'cats', 'q1', source=source_q3, line_color="black")

# Boxes
p_q3.vbar('cats', 0.7, 'q2', 'q3', source=source_q3, fill_color="#E6E6FA", line_color="black", legend_label="Boxplot")
p_q3.vbar('cats', 0.7, 'q1', 'q2', source=source_q3, fill_color="#6495ED", line_color="black", legend_label="Boxplot")

# Whiskers (horizontal lines at the ends of the whiskers)
p_q3.rect('cats', 'lower', 0.2, 0.01, source=source_q3, line_color="black")
p_q3.rect('cats', 'upper', 0.2, 0.01, source=source_q3, line_color="black")

# Outliers
outlier_renderer_q3 = p_q3.scatter('month', 'temp', size=6, color="red", alpha=0.6, source=outlier_source_q3, legend_label="Outliers")

hover_q3 = HoverTool(
    tooltips=[
        ("Month", "@cats"),
        ("Median", "@q2{0.0} °C"),
        ("Upper Quartile", "@q3{0.0} °C"),
        ("Lower Quartile", "@q1{0.0} °C"),
        ("Upper Whisker", "@upper{0.0} °C"),
        ("Lower Whisker", "@lower{0.0} °C")
    ],
    point_policy="follow_mouse"
)

outlier_hover_q3 = HoverTool(
    tooltips=[
        ("Month", "@month"),
        ("Temperature", "@temp{0.0} °C"),
        ("Status", "Outlier")
    ],
    renderers=[outlier_renderer_q3]
)

p_q3.add_tools(hover_q3)
p_q3.add_tools(outlier_hover_q3)

p_q3.xaxis.major_label_orientation = 45
p_q3.xgrid.grid_line_color = None
p_q3.ygrid.grid_line_color = "white"
p_q3.ygrid.grid_line_alpha = 0.9
p_q3.yaxis.formatter = NumeralTickFormatter(format="0.0")

p_q3.legend.location = "top_left"
p_q3.legend.click_policy = "hide"

show(p_q3)
print("\n" + "="*80 + "\n")

# ---
# ### Question 4: Yearly Box Plots with Color Mapping
#
# 4.  Create box plots to visualize the distribution of temperatures for each year,
#     and use color mapping to highlight temperature variations.
#
#     * Extract the **year** from the 'Date' column and create a new **'Year'** column.
#     * Group the data by **'Year'** and prepare it for plotting.
#     * Use Bokeh's box plot elements to visualize the distribution for each year.
#     * Label the x-axis with the **'Year'** and the y-axis with **"Temperature (°C)"**.
#     * Use `factor_cmap` to color the boxes based on the median temperature of each year.
#     * Add **tooltips** to display the year and relevant statistical values (min, max, median, etc.).
#     * Enable **pan, wheel zoom, and reset tools**.

print("--- Displaying Question 4: Yearly Box Plots with Color Mapping ---")
df['Year'] = df['Date'].dt.year.astype(str)

years_q4 = sorted(df['Year'].unique())

groups_q4 = df.groupby('Year')

cats_q4 = []
q1s_q4 = []
q2s_q4 = []
q3s_q4 = []
uppers_q4 = []
lowers_q4 = []
medians_q4 = []

for year in years_q4:
    group = groups_q4.get_group(year)
    q1 = group['Temperature'].quantile(0.25)
    q2 = group['Temperature'].quantile(0.5)
    q3 = group['Temperature'].quantile(0.75)
    iqr = q3 - q1
    upper = min(group['Temperature'].max(), q3 + 1.5 * iqr)
    lower = max(group['Temperature'].min(), q1 - 1.5 * iqr)

    cats_q4.append(year)
    q1s_q4.append(q1)
    q2s_q4.append(q2)
    q3s_q4.append(q3)
    uppers_q4.append(upper)
    lowers_q4.append(lower)
    medians_q4.append(q2)

source_q4 = ColumnDataSource(data=dict(
    cats=cats_q4,
    q1=q1s_q4,
    q2=q2s_q4,
    q3=q3s_q4,
    upper=uppers_q4,
    lower=lowers_q4,
    median=medians_q4
))

outliers_q4 = []
outlier_years_q4 = []

for year in years_q4:
    group = groups_q4.get_group(year)
    q1 = group['Temperature'].quantile(0.25)
    q3 = group['Temperature'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr

    outlier_idx = (group['Temperature'] > upper_bound) | (group['Temperature'] < lower_bound)
    year_outliers = group.loc[outlier_idx, 'Temperature'].tolist()
    outliers_q4.extend(year_outliers)
    outlier_years_q4.extend([year] * len(year_outliers))

outlier_source_q4 = ColumnDataSource(data=dict(
    year=outlier_years_q4,
    temp=outliers_q4
))

color_mapper_q4 = factor_cmap(
    field_name='cats',
    palette=Spectral11,
    factors=cats_q4
)

p_q4 = figure(
    title="Yearly Distribution of Minimum Temperatures",
    x_range=years_q4,
    x_axis_label="Year",
    y_axis_label="Temperature (°C)",
    height=500,
    width=800,
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

# Stems
p_q4.segment('cats', 'upper', 'cats', 'q3', source=source_q4, line_color="black")
p_q4.segment('cats', 'lower', 'cats', 'q1', source=source_q4, line_color="black")

# Boxes with color mapping
p_q4.vbar('cats', 0.7, 'q2', 'q3', source=source_q4, fill_color=color_mapper_q4, line_color="black")
p_q4.vbar('cats', 0.7, 'q1', 'q2', source=source_q4, fill_color=color_mapper_q4, line_color="black")

# Whiskers
p_q4.rect('cats', 'lower', 0.2, 0.01, source=source_q4, line_color="black")
p_q4.rect('cats', 'upper', 0.2, 0.01, source=source_q4, line_color="black")

# Outliers
outlier_renderer_q4 = p_q4.scatter('year', 'temp', size=6, color="red", alpha=0.6, source=outlier_source_q4)

hover_q4 = HoverTool(
    tooltips=[
        ("Year", "@cats"),
        ("Median", "@q2{0.0} °C"),
        ("Upper Quartile", "@q3{0.0} °C"),
        ("Lower Quartile", "@q1{0.0} °C"),
        ("Upper Whisker", "@upper{0.0} °C"),
        ("Lower Whisker", "@lower{0.0} °C")
    ],
    point_policy="follow_mouse"
)

outlier_hover_q4 = HoverTool(
    tooltips=[
        ("Year", "@year"),
        ("Temperature", "@temp{0.0} °C"),
        ("Status", "Outlier")
    ],
    renderers=[outlier_renderer_q4]
)

p_q4.add_tools(hover_q4)
p_q4.add_tools(outlier_hover_q4)

p_q4.xaxis.major_label_orientation = 45
p_q4.xgrid.grid_line_color = None
p_q4.ygrid.grid_line_color = "white"
p_q4.ygrid.grid_line_alpha = 0.9
p_q4.yaxis.formatter = NumeralTickFormatter(format="0.0")

show(p_q4)
print("\n" + "="*80 + "\n")

# ---
# ### Question 5: Interactive Time Range Selection
#
# 5.  Create an interactive line plot where the user can select a specific time range
#     to view using a date range slider.
#
#     * Create a basic line plot of **'Temperature'** over **'Date'**.
#     * Implement a date range slider using Bokeh widgets to allow users to select a start and end date.
#     * Update the plot dynamically based on the selected date range.
#     * Add **tooltips** to display the date and temperature.
#     * Enable **pan, wheel zoom, and reset tools**.

print("--- Displaying Question 5: Interactive Time Range Selection ---")
source_q5 = ColumnDataSource(df)

min_date_q5 = df['Date'].min()
max_date_q5 = df['Date'].max()

p_q5 = figure(
    title="Daily Minimum Temperatures - Interactive Time Range",
    x_axis_label="Date",
    y_axis_label="Temperature (°C)",
    x_axis_type="datetime",
    height=400,
    width=800,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    x_range=(min_date_q5, max_date_q5)
)

line_q5 = p_q5.line(
    x='Date',
    y='Temperature',
    source=source_q5,
    line_width=2,
    color='navy'
)

hover_q5 = HoverTool(
    tooltips=[
        ("Date", "@Date{%F}"),
        ("Temperature", "@Temperature{0.0} °C")
    ],
    formatters={
        "@Date": "datetime"
    },
    mode="vline"
)
p_q5.add_tools(hover_q5)

p_q5.xaxis.formatter = DatetimeTickFormatter(
    days="%d %b %Y",
    months="%b %Y",
    years="%Y"
)
p_q5.yaxis.formatter = NumeralTickFormatter(format="0.0")

select_q5 = figure(
    title="Drag the slider to select time range",
    height=130,
    width=800,
    y_range=p_q5.y_range,
    x_axis_type="datetime",
    tools="",
    toolbar_location=None,
    background_fill_color="#efefef"
)

select_q5.line(
    x='Date',
    y='Temperature',
    source=source_q5,
    color='navy'
)

range_tool_q5 = RangeTool(x_range=p_q5.x_range)
range_tool_q5.overlay.fill_color = "navy"
range_tool_q5.overlay.fill_alpha = 0.2
select_q5.add_tools(range_tool_q5)

select_q5.xaxis.formatter = DatetimeTickFormatter(
    days="%d %b %Y",
    months="%b %Y",
    years="%Y"
)
select_q5.yaxis.major_label_text_font_size = '0pt'

layout_q5 = column(p_q5, select_q5)

show(layout_q5)
print("\n" + "="*80 + "\n")

# ---
# ### Question 6: Time Series Decomposition Visualization
#
# 6.  Perform a simple time series decomposition to visualize the trend and seasonality
#     components of the temperature data.
#
#     * Resample the data to monthly frequency and calculate the monthly average temperature.
#     * Use a simple moving average to estimate the trend component.
#     * Calculate the seasonal component by subtracting the trend from the original monthly data.
#     * Create three separate Bokeh plots: one for the original monthly data, one for the trend,
#         and one for the seasonal component.
#     * Ensure the plots are aligned and share the same x-axis (Date).
#     * Add **tooltips** to each plot to display the date and corresponding value.
#     * Enable **pan, wheel zoom, and reset tools** for each plot.

print("--- Displaying Question 6: Time Series Decomposition Visualization ---")
# Resample to monthly frequency
monthly_df = df.set_index('Date').resample('M')['Temperature'].mean().reset_index()
monthly_df.columns = ['Date', 'Monthly_Avg']

# Calculate the trend component using a 12-month moving average
monthly_df['Trend'] = monthly_df['Monthly_Avg'].rolling(window=12, center=True).mean()

# Calculate the seasonal component
monthly_df['Seasonal'] = monthly_df['Monthly_Avg'] - monthly_df['Trend']

# Drop NA values after calculations
monthly_df = monthly_df.dropna().reset_index(drop=True)

source_q6 = ColumnDataSource(monthly_df)

# Original monthly data
p1_q6 = figure(
    title="Monthly Average Temperature (Original)",
    x_axis_label="",
    y_axis_label="Temperature (°C)",
    x_axis_type="datetime",
    height=250,
    width=800,
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

p1_q6.line(
    x='Date',
    y='Monthly_Avg',
    source=source_q6,
    line_width=2,
    color='navy'
)

# Trend component
p2_q6 = figure(
    title="Trend Component",
    x_axis_label="",
    y_axis_label="Temperature (°C)",
    x_axis_type="datetime",
    height=250,
    width=800,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    x_range=p1_q6.x_range
)

p2_q6.line(
    x='Date',
    y='Trend',
    source=source_q6,
    line_width=2,
    color='red'
)

# Seasonal component
p3_q6 = figure(
    title="Seasonal Component",
    x_axis_label="Date",
    y_axis_label="Temperature (°C)",
    x_axis_type="datetime",
    height=250,
    width=800,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    x_range=p1_q6.x_range
)

p3_q6.line(
    x='Date',
    y='Seasonal',
    source=source_q6,
    line_width=2,
    color='green'
)

# Add hover tools to each plot
for p_item, component_name in zip([p1_q6, p2_q6, p3_q6], ['Monthly_Avg', 'Trend', 'Seasonal']):
    hover_item = HoverTool(
        tooltips=[
            ("Date", "@Date{%F}"),
            (f"{component_name.replace('_', ' ')}", f"@{component_name}{{0.00}} °C")
        ],
        formatters={
            "@Date": "datetime"
        },
        mode="vline"
    )
    p_item.add_tools(hover_item)

    p_item.xaxis.formatter = DatetimeTickFormatter(
        days="%d %b %Y",
        months="%b %Y",
        years="%Y"
    )
    p_item.yaxis.formatter = NumeralTickFormatter(format="0.0")

# Hide x-axis labels and major ticks for upper plots for a cleaner stacked look
p1_q6.xaxis.major_label_text_font_size = '0pt'
p1_q6.xaxis.major_tick_line_color = None
p2_q6.xaxis.major_label_text_font_size = '0pt'
p2_q6.xaxis.major_tick_line_color = None

layout_q6 = column(p1_q6, p2_q6, p3_q6)

show(layout_q6)
print("\n" + "="*80 + "\n")

print("All Bokeh visualizations have been generated and displayed.")
```