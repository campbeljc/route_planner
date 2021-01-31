import numpy as np 
import geopandas as gpd 
import tkinter as tk
from tkinter import filedialog
import sys
import os

from load_gpxfile import *

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

test_file = '/Users/jennacampbell/Desktop/export_42781014/activities/4583421417.gpx'
series = convert_file_to_series(test_file)

#If predicting, create model that learns based on your data.
#Variables: day of week, time of day, time of year
#Output: type/length/elevation

#Find random route that matches type, length and elevation +/- 10%

#Map route as printout from geopandas + export shp file

#Potential Bonus: overlay on google maps