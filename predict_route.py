import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import geopandas as gpd
import datetime

#testing variables
gdf = gpd.read_file('test_file.json')
gdf.crs = 'epsg:5070'
gdf.to_crs('epsg:5070')
print(gdf.crs)
gdf['distance'] = gdf['geometry'].length
df = pd.DataFrame(gdf.drop(columns='geometry'))

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
    return df

def hot_encode(gdf):
    #get all time variables formatted and drop excess columns
    df_time = split_time(df).drop(columns='time')
    #get dummy variables for day column
    df_dummy_time = pd.get_dummies(df_time,columns=['day'])
    #get dummy variables for month
    df_dummy = pd.get_dummies(df_dummy_time,columns=['month'])
    return df_dummy

encoded_data = hot_encode(df)
print(encoded_data)

# def predict_type(gdf):

# def predict_bike(gdf):

# def predict_run(gdf):

# def predict_route(gdf):
    #run predict bike if type = bike or predict run if type = run