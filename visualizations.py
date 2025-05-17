#Import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_time_series(df):
    """
    Plot the temperatures and rainfall over time
    """
    # group by annual averages temperature and rainfall 

    annual_data = df.groupby("Year").mean().reset_index()

    fig, ax1 = plt.subplots(figsize=(14,8))

    # plot annual average temperature (left axis)
    ax1.plot(annual_data["Year"], annual_data["Average_Temperature_C"],
         'b', marker="o", label="Avg Temp")
    ax1.set_ylabel("Temperature (°C)", fontsize=14)
    ax1.set_xlabel("Year")
    ax1.tick_params(axis='y', labelcolor='b')

    # rainfall over the years (right axis)
    ax2 = ax1.twinx()
    ax2.plot(annual_data["Year"], annual_data["Total_Rainfall_mm"],
         'g', marker="o", label="Rainfall")
    ax2.set_ylabel("Rainfall (mm)", color='g', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='g')

    plt.title("Temperature vs Rainfall over the years (2000 to 2020)", fontsize=18)
    plt.grid(alpha=0.3)
    fig.legend(loc='upper left')
    return fig

def seasonal_temperature(df):
    """
    Identify seasonal patterns for temperature using decomposition techniques 
    """
    #creating the dataframe to datetime index
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
    df = df.set_index('date')

    # Make sure the data is sorted by date
    df = df.sort_index()
    climateseasonal=df.copy()
    climateseasonal = climateseasonal.drop(columns=['Year','Month'])

    # We analyze averege temperature)
    ts = climateseasonal['Average_Temperature_C']

    # Perform decomposition on the monthly data, use period=12 (12 months in a year)
    result = seasonal_decompose(ts, model='additive', period=12)
    # plot components individually 
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig1.suptitle("Seasonal patterns of monthly Average Temperature")
    result.observed.plot(ax=ax1, title='Observed')
    result.trend.plot(ax=ax2, title='Trend')
    result.seasonal.plot(ax=ax3, title='Seasonal')
    result.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()


    # Extract the seasonal component
    seasonal_component = result.seasonal

    # Get the average seasonal pattern by month
    monthly_seasonal = seasonal_component.groupby(seasonal_component.index.month).mean()

    # Plot the seasonal pattern
    fig2, ax5 = plt.subplots(figsize=(10, 4))
    # Plot the seasonal pattern using the axis object
    monthly_seasonal.plot(kind='bar', ax=ax5)
    ax5.set_title('Average Seasonal Pattern for Temperature by Month')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Seasonal Component')
    #Set x-ticks using the axis object
    ax5.set_xticks(ticks=range(12))
    ax5.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    #add grid using the axis object
    ax5.grid(alpha=0.3)
    plt.close('all')

    return fig1, fig2

def seasonal_rainfall(df):
    """
    Identify seasonal patterns for rainfall using decomposition techniques 
    """
    #creating the dataframe to datetime index
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
    df = df.set_index('date')

    # Make sure the data is sorted by date
    df = df.sort_index()
    climateseasonal=df.copy()
    climateseasonal = climateseasonal.drop(columns=['Year','Month'])

    # We analyze total rainfall
    ts2 = climateseasonal['Total_Rainfall_mm']

    # Perform decomposition on the monthly data, use period=12 (12 months in a year)
    result2 = seasonal_decompose(ts2, model='additive', period=12)
    # Plotting components individually 
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig1.suptitle("Seasonal patterns of monthly Total Rainfall")
    result2.observed.plot(ax=ax1, title='Observed')
    result2.trend.plot(ax=ax2, title='Trend')
    result2.seasonal.plot(ax=ax3, title='Seasonal')
    result2.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()

    # Extract the seasonal component
    seasonal_component2 = result2.seasonal

    # Get the average seasonal pattern by month
    monthly_seasonal = seasonal_component2.groupby(seasonal_component2.index.month).mean()

    # Plot the seasonal pattern
    fig2, ax5 = plt.subplots(figsize=(10, 4))
    
    monthly_seasonal.plot(kind='bar', ax=ax5)
    ax5.set_title('Average Seasonal Pattern for Rainfall by Month')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Seasonal Component')
    ax5.set_xticks(ticks=range(12))
    ax5.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax5.grid(alpha=0.3)
    plt.close('all')
    return fig1, fig2

def heatmap_corr(df):
    """
    Heatmap indicating correlation between variables    
    """
    #correlation between variables

    annual_data = df.groupby("Year").mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10,8))
    numeric_variables = annual_data[['Average_Temperature_C', 'Total_Rainfall_mm','Max_Temperature_C', 'Min_Temperature_C']]

    sns.heatmap(numeric_variables.corr(), 
            cmap='coolwarm', 
            annot=True, 
            vmin=-1, 
            vmax=1, 
            ax=ax)
    ax.set_title("Correlation Heatmap: Temperature and Rainfall over 21 Years")
    plt.tight_layout()
    return fig 
