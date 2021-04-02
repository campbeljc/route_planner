import numpy as np 
import geopandas as gpd 
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import sys
import os
import datetime

from load_gpxfile import create_df
from predict_route import create_model, hot_encode, analyze_activitymodel, create_metrics_model, get_error, split_time, add_all_daysmonths, add_all_activities

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

    #create model to predict type of activity
    model, X_test, y_test, cross_val = create_model(encoded_cleaned)
    proportion_error, matrix, f1  = analyze_activitymodel(model, X_test, y_test)

    #create model to predict distance and elevation
    model_met, X_test_met, y_test_met, cross_val_met = create_metrics_model(encoded_cleaned)
    actual_results_met, predicted_results_met, error_met= get_error(model_met, X_test_met, y_test_met)

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
    current_cleaned = current_encoded.iloc[:-19]
    
    #predict current activity
    predicted_activity = model.predict(current_cleaned)
    
    #predict distance and elevation
    current_cleaned['activity'] = float(predicted_activity[0])
    # print(predicted_activity[0])

    #add all possible activities for use in hot encoding
    all_activities = add_all_activities(current_cleaned)
    # print(all_activities['activity'])

    dummy_activity = pd.get_dummies(all_activities,columns=['activity'])
    activity_cleaned = dummy_activity.iloc[:-2]

    #predict metrics using same data with model_met
    predicted_metrics = model_met.predict(activity_cleaned)
    print(predicted_metrics)

#Find random route that matches type, length and elevation +/- 10%

#Map route as printout from geopandas + export shp file

#Potential Bonus: overlay on google maps