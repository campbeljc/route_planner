import numpy as np
import pandas as pd
import geopandas as gpd
import datetime

#Load in test dataset
# gdf = gpd.read_file('my_gdf.json')

def split_time(df):
    #Format times as datetime objects and subtract 8 hours (for time difference?)
    time_series = df['time']
    time_cleaned = [x[:-6] for x in time_series]
    df['datetime'] = [datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S') for x in time_cleaned]
    df['datetime'] = [x - datetime.timedelta(hours=8) for x in df['datetime']]
    #Find day of week for each row
    df['day'] = [x.weekday() for x in df['datetime']]
    #Find seconds since midnight for every row
    df['time_of_day'] = [datetime.timedelta(hours=t.hour, minutes=t.minute,seconds=t.second).total_seconds() for t in df['datetime']]
    #Find month
    df['month'] = [x.month for x in df['datetime']]
    df = df.drop(columns='time')
    return df

def add_all_daysmonths(df):
    num_columns = len(df.columns)
    blank_series = pd.Series((np.nan for x in np.arange(num_columns)),index=df.columns)
    for month in np.arange(1,13):
        blank_series['month'] = month
        df = df.append(blank_series, ignore_index=True)
    for day in np.arange(1,8):
        blank_series['day'] = day
        df = df.append(blank_series, ignore_index=True)
    return df

def hot_encode(df):
    #get dummy variables for day column
    df_dummy_time = pd.get_dummies(df,columns=['day'])
    #get dummy variables for month
    df_dummy = pd.get_dummies(df_dummy_time,columns=['month'])
    return df_dummy

def add_all_activities(df):
    num_columns = len(df.columns)
    blank_series = pd.Series((np.nan for x in np.arange(num_columns)),index=df.columns)
    for activity in np.array([1,9]):
        string_activity = str(activity)
        blank_series['activity'] = string_activity
        df = df.append(blank_series, ignore_index=True)
    return df