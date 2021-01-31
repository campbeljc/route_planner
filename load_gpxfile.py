#Create Directory: Df of all activity information. Includes, Type of route, elevation, length and gdf with info
#For each activity, convert to geodataframe and store in larger dataframe

import os
import numpy as np 
import pandas as pd 
import fiona
import geopandas as gpd

test_file = '/Users/jennacampbell/Desktop/export_42781014/activities/4583421417.gpx'
test_dir = '/Users/jennacampbell/Desktop/export_42781014/activities'

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

def convert_file_to_series(fname):
    time, lon, lat, activity, elevation_list = collect_data_from_gpx(fname)
    elevation = calculate_total_elevation(elevation_list)
    return pd.Series([time, lon, lat, activity, elevation,])

def create_df(dir_path):
    df = pd.DataFrame([],columns=['activity','time','elevation','lon','lat'])
    count = 1
    for filename in os.listdir(dir_path):
        if filename.endswith(".gpx"):
            print("file #" + str(count) + " - " + filename)
            file_path = os.path.join(dir_path, filename)
            series = convert_file_to_series(file_path)
            df = df.append(series, ignore_index=True)
            count += 1
    return df

df = create_df(test_dir)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
gdf.crs = 'epsg:4326'

# gdf.to_file('my_gdf.json', driver='GeoJSON')
gdf.to_file("output.gpkg", driver="GPKG")