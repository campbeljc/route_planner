import geopandas as gpd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors

#Load in test dataset
gdf = gpd.read_file('my_gdf.json')

#function that predicts activity based on nearest neghbors (KNeighborsClassifier)
def create_model(df):
    #create X data and y data
    X = df.drop(['activity','elevation','distance','datetime','geometry'],axis=1)
    y = np.array(df[['activity']]).reshape(1,-1)[0]
    #create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    #create model
    classifier = KNeighborsClassifier(n_neighbors=4, algorithm="kd_tree")
    #to fine tune algorithm use algorithm = "BallTree" or "brute"
    model = classifier.fit(X_train, y_train)
    return model


#funciton that predicts distance based on nearest neighbors (KNeighborsRegressor)
def create_metrics_model(df):
    #get X and y variables. X should now include activity.
    X = df.drop(['elevation','distance','datetime','geometry'],axis=1)
    y = df[['distance','elevation']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    regressor = KNeighborsRegressor(n_neighbors=4, algorithm="kd_tree")
    model = regressor.fit(X_train, y_train)
    return model

#function that selects nearest neighbor to select route. Chooses based off activity, elevation and distance (NearestNeighbors)
def find_closest_route(df, activity, distance, elevation):
    X = df[['activity','elevation','distance']].astype(np.float64)
    neighbors = NearestNeighbors(algorithm="kd_tree")
    neighbors.fit(X)
    sample = np.array([activity, distance, elevation]).reshape(1,-1)
    index = neighbors.kneighbors(sample, n_neighbors=1, return_distance=False)
    series = df.iloc[index[0]]

    return series