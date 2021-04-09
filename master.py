import numpy as np 
import geopandas as gpd 
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import sys
import os
import datetime
import matplotlib.pyplot as plt
import contextily as ctx
import fiona

from load_gpxfile import create_df
from support_functions import hot_encode, split_time, add_all_daysmonths, add_all_activities
from neighbors_classifier import create_model, create_metrics_model, find_closest_route

##Outline
#Initializing
# Ask for:
# 1) Zip file of Strava Data
# 2) Predict Route
# If not predicting route ask for type of route and length. If bike ask for elevation
testing=sys.argv[1]
if testing == "True":
    file_path = "/Users/jennacampbell/Desktop/export_42781014/activities"
    activity = "Bike"
    distance = 30
    elevation = 2000
    predict = 'yes'
else:
    distance = np.nan
    activity = np.nan
    elevation = np.nan

    predict = input ("Predict route? yes or no: ")

    while not(predict in ['No','no','yes','Yes']):
        predict = input("Please input yes or no: ")

    if predict == 'no' or predict == 'No':
        activity = input("Enter Activity (bike or run): ")
        while not(activity in ['Bike','bike','Run','run']):
            activity= input("Please input bike or run: ")
        distance = input("Enter Distance: ")
        try:
            int(distance)
        except:
            distance = input("Please enter a number: ")
        if activity == "bike" or activity == "Bike":
            elevation = input("Enter Elevation: ")
            try:
                int(elevation)
            except:
                elevation = input("Please enter a number: ")

    # Read in zip file and navigate to activities directory
    print("Got it! Now please select the folder containing your downloaded Strava data")

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askdirectory()
    file_path = os.path.join(file_path, "activities")

if testing == "True":
    gdf = gpd.read_file('test_file.json')
else:
    #Create GeoDataFrame out of DataFrame
    df = create_df(file_path)

    gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=df['coordinates'])
    gdf = gdf.drop(['coordinates'],axis=1)
    gdf = gdf.to_crs('epsg:5070') 
    gdf.crs = 'epsg:5070'
    #Add distance column and adjust activities to only include biking and running

gdf['distance'] = gdf['geometry'].length
gdf = gdf[(gdf['activity'] == "1") | (gdf['activity'] == "9")]

#If predicting, create model that learns based on your data.
#Variables: day of week, time of day, time of year
#Output: type/length/elevation

if predict == 'yes' or predict == 'Yes':
    #convert datetime data to numerical data
    df_time = split_time(gdf)

    #add all possible days/months
    df_all = add_all_daysmonths(df_time)
    
    encoded_data = hot_encode(df_all)
    #remove all extra columns that were added to hot encode with all possible months/days
    encoded_cleaned = encoded_data.iloc[:-19]


    #create model to predict activity using encoded_cleaned data
    #create model to predict distance and elevation using encoded_cleaned data with predicted activity added
    #create model to predict type of activity
    model = create_model(encoded_cleaned)
    #get dummies for activity

    # encoded_cleaned['activity'] = encoded_cleaned['activity'].astype(np.float64)
    pd.set_option('mode.chained_assignment',None)
    encoded_cleaned.loc[:,'activity'] = encoded_cleaned['activity'].astype(np.float64)
    #^^ this line causes the SettingwithCopyWarning

    encoded_activity = pd.get_dummies(encoded_cleaned,columns=['activity'])

    #create model to predict distance and elevation
    model_met = create_metrics_model(encoded_activity)

    #get current time and create dataframe in same format as gdf to be inserted into hot_encode
    today = datetime.datetime.now()
    day = today.weekday()
    time_of_day = datetime.timedelta(hours=today.hour, minutes=today.minute,seconds=today.second).total_seconds()
    month = today.month
    current_df = pd.DataFrame([[day, time_of_day, month]],columns=['day', 'time_of_day','month'])

    #add all days/months to get all possible columns when hot encoding
    current_all = add_all_daysmonths(current_df)
    current_encoded = hot_encode(current_all)
    #remove all excess columns used for hot encode
    current_cleaned = current_encoded.iloc[:-19].astype(np.float64)
    
    #predict current activity
    predicted_activity = model.predict(current_cleaned)
    
    #predict distance and elevation
    current_cleaned['activity'] = float(predicted_activity[0])

    #add all possible activities for use in hot encoding
    all_activities = add_all_activities(current_cleaned)

    dummy_activity = pd.get_dummies(all_activities,columns=['activity'])
    activity_cleaned = dummy_activity.iloc[:-2].astype(np.float64)

    #predict metrics using same data with model_met
    predicted_metrics = model_met.predict(activity_cleaned)

    #set predicted activiy and metrics to appropriate variables
    activity = predicted_activity[0]
    distance = predicted_metrics[0][0]
    elevation = predicted_metrics[0][1]

#Find route that is closest to given activity, elevation and distance. Returns series with route info including geometry
route = find_closest_route(gdf, activity, distance, elevation)
route = route.to_crs(3857)
route = route.set_crs('epsg:3857')

minx, miny, maxx, maxy = route.geometry.total_bounds

#Map route as printout from geopandas

fig, ax = plt.subplots(1, 1)
# world.plot(ax=ax)
route.plot(ax=ax, column='elevation')

ax.set_xlim(minx - 200, maxx + 200)
ax.set_ylim(miny - 200, maxy + 200)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# plt.show()

#Export file as kml file
fiona.supported_drivers['KML'] = 'rw'
route.to_file('test.kml', driver='KML')

#Potential Bonus: overlay on google maps

#TODO
#1. Save file and image to folder entitled "suggested routes" in strava route data
#2. Add some sort of accuracy check to learning algorithms and adjust to maximize this metric