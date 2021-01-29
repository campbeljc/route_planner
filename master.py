import numpy as np 
import geopandas as gpd 

##Outline
#Initializing
# Ask for:
# 1) Zip file of Strava Data
# 2) Predict Route
# If not predicting route ask for type of route and length. If bike ask for elevation

# Read in zip file and navigate to activities directory


#Create Directory: Df of all activity information. Includes, Type of route, elevation, length and gdf with info
#For each activity, convert to geodataframe and store in larger dataframe

#If predicting, create model that learns based on your data.
#Variables: day of week, time of day, time of year
#Output: type/length/elevation

#Find random route that matches type, length and elevation +/- 10%

#Map route as printout from geopandas + export shp file

#Potential Bonus: overlay on google maps