import numpy as np
import pandas as pd


#  Make prediction

def make_prediction(model, year, month):
    """
    Make prediction of temperature for a given month and year
    """

    features = np.array([year, month])

    return model.predict(features)[0]

# get historical contenxt
def get_historical_context(df, month):
    """
    Get the historical context
    """

    years = df['year'].unique()
    hist_temps = []

    for year in years:
        month_data = df[(df['year'] == year) & (df['month']== month)]
        if not month_data.empty:
            hist_temps.append((year, month_data['temperature'].values[0]))


    return hist_temps

def get_historical_average(df, month):
    """
    Get historical average temp for a given month
    """

    return df[df['month'] == month]['temperature'].mean()