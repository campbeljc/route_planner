#Create Directory: Df of all activity information. Includes, Type of route, elevation, length and gdf with info
#For each activity, convert to geodataframe and store in larger dataframe

import os
import numpy as np 
import geopandas as gpd 
import fiona

test_file = '/Users/jennacampbell/Desktop/export_42781014/activities/4583421417.gpx'

def collect_data_from_gpx(fname):
    #selects gpx layer containing time variable and extract time
    track_points = fiona.open(fname, layer='track_points')
    time = track_points[0]['properties']['time']
    #selects gpx layer containing geometry variables and extracts geometry and type of activity
    layer = fiona.open(fname, layer='tracks')
    geom = layer[0]
    coordinates = geom['geometry']['coordinates']
    activity = geom['properties']['type']
    #Calculate total
    elevation = []
    for entry in track_points:
        elevation = elevation + [entry['properties']['ele']]
    #unzip coordinate data
    lon, lat = zip(*coordinates[0])
    return time, lon, lat, activity, elevation

def calculate_total_elevation(elevation):
    total = 0
    previous = elevation[0]
    for ele in elevation:
        if ele - previous >= 0:
            total = total + (ele - previous)
        previous = ele
    #3.28 is to convert meters to ft
    return total*3.28

time, lon, lat, activity, elevation = collect_data_from_gpx(test_file)
elevation = calculate_total_elevation(elevation_list)

def convert_to_df(time, lon, lat, activity, elevation):


# for filename in os.listdir(file_path):
#     if filename.endswith(".gpx"):
        