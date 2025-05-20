import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    DatetimeTickFormatter,
    NumeralTickFormatter,
    BoxAnnotation,
    DataRange1d,
    RangeTool,
    DateRangeSlider,
    CustomJS
)
from bokeh.layouts import row, column, gridplot
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral11, Turbo256
import numpy as np
from statistics import median

output_notebook()  # Enable Bokeh output in Jupyter Notebook

# Load the Dataset - Ajuster le chemin selon votre structure de dossiers
try:
    # Tentative avec le chemin relatif standard
    df = pd.read_csv("daily-minimum-temperatures-in-melbourne.csv")
except FileNotFoundError:
    try:
        # Tentative avec le chemin incluant le dossier datasets
        df = pd.read_csv("lab-sessions\datasets\daily-minimum-temperatures-in-melbourne.csv")
    except FileNotFoundError:
        # Si vous utilisez ce code, vous devrez ajuster le chemin selon votre environnement
        df = pd.read_csv("C:/Users/victo/Documents/GitHub/Data_Visualization_Course/Data_Visualization_Course/datasets/daily-minimum-temperatures-in-melbourne.csv")

# Rename columns for clarity
df.columns = ['Date', 'Temperature']

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Remove '?' from the 'Temperature' column and convert to numeric
df['Temperature'] = df['Temperature'].astype(str).str.replace('?', '', regex=False)
df['Temperature'] = pd.to_numeric(df['Temperature'])

# Question 1: Basic Time Series Line Plot
def create_basic_line_plot():
    # Create a ColumnDataSource
    source = ColumnDataSource(df)
    
    # Create the figure
    p = figure(
        title="Daily Minimum Temperatures",
        x_axis_label="Date",
        y_axis_label="Temperature (°C)",
        x_axis_type="datetime",
        height=400,
        width=800,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Add the line
    p.line(
        x='Date',
        y='Temperature',
        source=source,
        line_width=2,
        color='navy'
    )
    
    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ("Date", "@Date{%F}"),
            ("Temperature", "@Temperature{0.0} °C")
        ],
        formatters={
            "@Date": "datetime"
        },
        mode="vline"
    )
    p.add_tools(hover)
    
    # Format axes
    p.xaxis.formatter = DatetimeTickFormatter(
        days="%d %b %Y",
        months="%b %Y",
        years="%Y"
    )
    p.yaxis.formatter = NumeralTickFormatter(format="0.0")
    
    return p

# Question 2: Rolling Average
def create_rolling_average_plot():
    # Calculate the 30-day rolling average
    df['Rolling_Avg'] = df['Temperature'].rolling(window=30).mean()
    
    # Create a ColumnDataSource
    source = ColumnDataSource(df)
    
    # Create the figure
    p = figure(
        title="Daily Minimum Temperatures with 30-Day Rolling Average",
        x_axis_label="Date",
        y_axis_label="Temperature (°C)",
        x_axis_type="datetime",
        height=400,
        width=800,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Add the original temperature line
    p.line(
        x='Date',
        y='Temperature',
        source=source,
        line_width=1,
        color='navy',
        alpha=0.7,
        legend_label="Daily Temperature"
    )
    
    # Add the rolling average line
    p.line(
        x='Date',
        y='Rolling_Avg',
        source=source,
        line_width=2,
        color='red',
        line_dash='solid',
        legend_label="30-Day Rolling Average"
    )
    
    # Add hover tool
    hover = HoverTool(
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
    p.add_tools(hover)
    
    # Format axes
    p.xaxis.formatter = DatetimeTickFormatter(
        days="%d %b %Y",
        months="%b %Y",
        years="%Y"
    )
    p.yaxis.formatter = NumeralTickFormatter(format="0.0")
    
    # Configure legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p

# Question 3: Monthly Box Plots
def create_monthly_box_plots():
    # Extract month from Date and create a new Month column
    df['Month'] = df['Date'].dt.month_name()
    
    # Define the correct order of months
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    
    # Prepare data for box plots
    groups = df.groupby('Month')
    
    # Lists to store data for box plots
    cats = []
    q1s = []
    q2s = []
    q3s = []
    uppers = []
    lowers = []
    
    # Calculate the statistics for each month
    for month_name in month_order:
        group = groups.get_group(month_name)
        q1 = group['Temperature'].quantile(0.25)
        q2 = group['Temperature'].quantile(0.5)
        q3 = group['Temperature'].quantile(0.75)
        iqr = q3 - q1
        upper = min(group['Temperature'].max(), q3 + 1.5 * iqr)
        lower = max(group['Temperature'].min(), q1 - 1.5 * iqr)
        
        cats.append(month_name)
        q1s.append(q1)
        q2s.append(q2)
        q3s.append(q3)
        uppers.append(upper)
        lowers.append(lower)
    
    # Create a ColumnDataSource for the box plot data
    source = ColumnDataSource(data=dict(
        cats=cats,
        q1=q1s,
        q2=q2s,
        q3=q3s,
        upper=uppers,
        lower=lowers
    ))
    
    # Create additional data for outliers
    outliers = []
    outlier_months = []
    
    for month_name in month_order:
        group = groups.get_group(month_name)
        q1 = group['Temperature'].quantile(0.25)
        q3 = group['Temperature'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        
        outlier_idx = (group['Temperature'] > upper_bound) | (group['Temperature'] < lower_bound)
        month_outliers = group.loc[outlier_idx, 'Temperature'].tolist()
        outliers.extend(month_outliers)
        outlier_months.extend([month_name] * len(month_outliers))
    
    outlier_source = ColumnDataSource(data=dict(
        month=outlier_months,
        temp=outliers
    ))
    
    # Create the figure
    p = figure(
        title="Monthly Distribution of Minimum Temperatures",
        x_range=month_order,
        x_axis_label="Month",
        y_axis_label="Temperature (°C)",
        height=500,
        width=800,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Add the box glyphs
    # Stems
    p.segment('cats', 'upper', 'cats', 'q3', source=source, line_color="black")
    p.segment('cats', 'lower', 'cats', 'q1', source=source, line_color="black")
    
    # Boxes
    p.vbar('cats', 0.7, 'q2', 'q3', source=source, fill_color="#E6E6FA", line_color="black", legend_label="Boxplot")
    p.vbar('cats', 0.7, 'q1', 'q2', source=source, fill_color="#6495ED", line_color="black", legend_label="Boxplot")
    
    # Whiskers (almost-0 height rects)
    p.rect('cats', 'lower', 0.2, 0.01, source=source, line_color="black")
    p.rect('cats', 'upper', 0.2, 0.01, source=source, line_color="black")
    
    # Outliers
    outlier_renderer = p.scatter('month', 'temp', size=6, color="red", alpha=0.6, source=outlier_source, legend_label="Outliers")
    
    # Add hover tool
    hover = HoverTool(
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
    
    outlier_hover = HoverTool(
        tooltips=[
            ("Month", "@month"),
            ("Temperature", "@temp{0.0} °C"),
            ("Status", "Outlier")
        ],
        renderers=[outlier_renderer]
    )
    
    p.add_tools(hover)
    p.add_tools(outlier_hover)
    
    # Format the plot
    p.xaxis.major_label_orientation = 45
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "white"
    p.ygrid.grid_line_alpha = 0.9
    p.yaxis.formatter = NumeralTickFormatter(format="0.0")
    
    # Configure legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p

# Question 4: Yearly Box Plots with Color Mapping
def create_yearly_box_plots():
    # Extract year from Date and create a new Year column
    df['Year'] = df['Date'].dt.year.astype(str)
    
    # Get the unique years in the dataset
    years = sorted(df['Year'].unique())
    
    # Group by Year for statistics
    groups = df.groupby('Year')
    
    # Lists to store data for box plots
    cats = []
    q1s = []
    q2s = []
    q3s = []
    uppers = []
    lowers = []
    medians = []  # For color mapping
    
    # Calculate the statistics for each year
    for year in years:
        group = groups.get_group(year)
        q1 = group['Temperature'].quantile(0.25)
        q2 = group['Temperature'].quantile(0.5)
        q3 = group['Temperature'].quantile(0.75)
        iqr = q3 - q1
        upper = min(group['Temperature'].max(), q3 + 1.5 * iqr)
        lower = max(group['Temperature'].min(), q1 - 1.5 * iqr)
        
        cats.append(year)
        q1s.append(q1)
        q2s.append(q2)
        q3s.append(q3)
        uppers.append(upper)
        lowers.append(lower)
        medians.append(q2)  # Store median for color mapping
    
    # Create a ColumnDataSource for the box plot data
    source = ColumnDataSource(data=dict(
        cats=cats,
        q1=q1s,
        q2=q2s,
        q3=q3s,
        upper=uppers,
        lower=lowers,
        median=medians
    ))
    
    # Create additional data for outliers
    outliers = []
    outlier_years = []
    
    for year in years:
        group = groups.get_group(year)
        q1 = group['Temperature'].quantile(0.25)
        q3 = group['Temperature'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        
        outlier_idx = (group['Temperature'] > upper_bound) | (group['Temperature'] < lower_bound)
        year_outliers = group.loc[outlier_idx, 'Temperature'].tolist()
        outliers.extend(year_outliers)
        outlier_years.extend([year] * len(year_outliers))
    
    outlier_source = ColumnDataSource(data=dict(
        year=outlier_years,
        temp=outliers
    ))
    
    # Define color palette based on median temperature
    color_mapper = factor_cmap(
        field_name='cats',
        palette=Spectral11,
        factors=cats
    )
    
    # Create the figure
    p = figure(
        title="Yearly Distribution of Minimum Temperatures",
        x_range=years,
        x_axis_label="Year",
        y_axis_label="Temperature (°C)",
        height=500,
        width=800,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Add the box glyphs
    # Stems
    p.segment('cats', 'upper', 'cats', 'q3', source=source, line_color="black")
    p.segment('cats', 'lower', 'cats', 'q1', source=source, line_color="black")
    
    # Boxes with color mapping
    p.vbar('cats', 0.7, 'q2', 'q3', source=source, fill_color=color_mapper, line_color="black")
    p.vbar('cats', 0.7, 'q1', 'q2', source=source, fill_color=color_mapper, line_color="black")
    
    # Whiskers
    p.rect('cats', 'lower', 0.2, 0.01, source=source, line_color="black")
    p.rect('cats', 'upper', 0.2, 0.01, source=source, line_color="black")
    
    # Outliers
    outlier_renderer = p.scatter('year', 'temp', size=6, color="red", alpha=0.6, source=outlier_source)
    
    # Add hover tool
    hover = HoverTool(
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
    
    outlier_hover = HoverTool(
        tooltips=[
            ("Year", "@year"),
            ("Temperature", "@temp{0.0} °C"),
            ("Status", "Outlier")
        ],
        renderers=[outlier_renderer]
    )
    
    p.add_tools(hover)
    p.add_tools(outlier_hover)
    
    # Format the plot
    p.xaxis.major_label_orientation = 45
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "white"
    p.ygrid.grid_line_alpha = 0.9
    p.yaxis.formatter = NumeralTickFormatter(format="0.0")
    
    return p

# Question 5: Interactive Time Range Selection
def create_interactive_time_range_plot():
    # Create a ColumnDataSource
    source = ColumnDataSource(df)
    
    # Define the date range for the slider
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    # Create the main figure
    p = figure(
        title="Daily Minimum Temperatures - Interactive Time Range",
        x_axis_label="Date",
        y_axis_label="Temperature (°C)",
        x_axis_type="datetime",
        height=400,
        width=800,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_range=(min_date, max_date)
    )
    
    # Add the temperature line
    line = p.line(
        x='Date',
        y='Temperature',
        source=source,
        line_width=2,
        color='navy'
    )
    
    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ("Date", "@Date{%F}"),
            ("Temperature", "@Temperature{0.0} °C")
        ],
        formatters={
            "@Date": "datetime"
        },
        mode="vline"
    )
    p.add_tools(hover)
    
    # Format axes
    p.xaxis.formatter = DatetimeTickFormatter(
        days="%d %b %Y",
        months="%b %Y",
        years="%Y"
    )
    p.yaxis.formatter = NumeralTickFormatter(format="0.0")
    
    # Create a secondary range tool plot
    select = figure(
        title="Drag the slider to select time range",
        height=130,
        width=800,
        y_range=p.y_range,
        x_axis_type="datetime",
        tools="",
        toolbar_location=None,
        background_fill_color="#efefef"
    )
    
    # Add a line renderer to the range tool plot
    select.line(
        x='Date',
        y='Temperature',
        source=source,
        color='navy'
    )
    
    # Add the range tool
    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select.add_tools(range_tool)
    
    # Format axes for the range plot
    select.xaxis.formatter = DatetimeTickFormatter(
        days="%d %b %Y",
        months="%b %Y",
        years="%Y"
    )
    select.yaxis.major_label_text_font_size = '0pt'  # Hide y-axis labels
    
    # Combine the plots
    layout = column(p, select)
    
    return layout

# Question 6: Time Series Decomposition Visualization
def create_decomposition_visualization():
    # Resample to monthly frequency
    monthly_df = df.set_index('Date').resample('M')['Temperature'].mean().reset_index()
    monthly_df.columns = ['Date', 'Monthly_Avg']
    
    # Calculate the trend component using a 12-month moving average
    monthly_df['Trend'] = monthly_df['Monthly_Avg'].rolling(window=12, center=True).mean()
    
    # Calculate the seasonal component
    monthly_df['Seasonal'] = monthly_df['Monthly_Avg'] - monthly_df['Trend']
    
    # Drop NA values after calculations
    monthly_df = monthly_df.dropna()
    
    # Create a ColumnDataSource
    source = ColumnDataSource(monthly_df)
    
    # Create figures for each component
    
    # Original monthly data
    p1 = figure(
        title="Monthly Average Temperature",
        x_axis_label="",  # Will be added to the bottom plot only
        y_axis_label="Temperature (°C)",
        x_axis_type="datetime",
        height=250,
        width=800,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    p1.line(
        x='Date',
        y='Monthly_Avg',
        source=source,
        line_width=2,
        color='navy'
    )
    
    # Trend component
    p2 = figure(
        title="Trend Component",
        x_axis_label="",  # Will be added to the bottom plot only
        y_axis_label="Temperature (°C)",
        x_axis_type="datetime",
        height=250,
        width=800,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_range=p1.x_range  # Share the x-range
    )
    
    p2.line(
        x='Date',
        y='Trend',
        source=source,
        line_width=2,
        color='red'
    )
    
    # Seasonal component
    p3 = figure(
        title="Seasonal Component",
        x_axis_label="Date",
        y_axis_label="Temperature (°C)",
        x_axis_type="datetime",
        height=250,
        width=800,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_range=p1.x_range  # Share the x-range
    )
    
    p3.line(
        x='Date',
        y='Seasonal',
        source=source,
        line_width=2,
        color='green'
    )
    
    # Add hover tools to each plot
    for p, component in zip([p1, p2, p3], ['Monthly_Avg', 'Trend', 'Seasonal']):
        hover = HoverTool(
            tooltips=[
                ("Date", "@Date{%F}"),
                ("Value", f"@{component}{{0.00}} °C")
            ],
            formatters={
                "@Date": "datetime"
            },
            mode="vline"
        )
        p.add_tools(hover)
        
        # Format axes
        p.xaxis.formatter = DatetimeTickFormatter(
            days="%d %b %Y",
            months="%b %Y",
            years="%Y"
        )
        p.yaxis.formatter = NumeralTickFormatter(format="0.0")
    
    # Only show the date axis on the bottom plot
    p1.xaxis.visible = False
    p2.xaxis.visible = False
    
    # Combine the plots in a column
    layout = column(p1, p2, p3)
    
    return layout

# Display all visualizations
# Question 1
q1_plot = create_basic_line_plot()
show(q1_plot)

# Question 2
q2_plot = create_rolling_average_plot()
show(q2_plot)

# Question 3
q3_plot = create_monthly_box_plots()
show(q3_plot)

# Question 4
q4_plot = create_yearly_box_plots()
show(q4_plot)

# Question 5
q5_plot = create_interactive_time_range_plot()
show(q5_plot)

# Question 6
q6_plot = create_decomposition_visualization()
show(q6_plot)