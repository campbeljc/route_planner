import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
import geopandas as gpd
import datetime

#Load in test dataset
# gdf = gpd.read_file('my_gdf.json')

#Calculate distance variable
# gdf['distance'] = gdf['geometry'].length
# gdf = gdf[(gdf['activity'] == '1') | (gdf['activity'] == '9')]
# df = pd.DataFrame(gdf.drop(columns='geometry'))

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
    # average_tod = df['time_of_day'].mean()
    # df['time_of_day'] = df['time_of_day']/average_tod
    #Find month
    df['month'] = [x.month for x in df['datetime']]
    return df

def hot_encode(df):
    #get all time variables formatted and drop excess columns
    df_time = split_time(df).drop(columns='time')
    #get dummy variables for day column
    df_dummy_time = pd.get_dummies(df_time,columns=['day'])
    #get dummy variables for month
    df_dummy = pd.get_dummies(df_dummy_time,columns=['month'])
    return df_dummy

# encoded_data = hot_encode(df)

def create_model(df):
    #Create training and testing datasets
    X = df.drop(['activity','elevation','distance','datetime','geometry'],axis=1)
    y = np.array(df[['activity']]).reshape(1,-1)[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    # # model = LogisticRegression(C=10000, class_weight='balanced').fit(X_train, y_train)
    # # model = GaussianNB().fit(X_train, y_train)
    classifier = MLPClassifier(hidden_layer_sizes=(12,12,12),max_iter=2000)
    model = classifier.fit(X_train, y_train)
    cross_val = cross_val_score(classifier, X, y, cv=3)
    return model, X_test, y_test, cross_val

# model, X_test, y_test, cross_val = create_model(encoded_data)

def analyze_activitymodel(model, X_test, y_test):
    predicted_activity = pd.Series(model.predict(X_test))
    predicted_zeros = predicted_activity.replace('1',1).replace('9',0)
    actual_results = pd.Series(y_test)
    actual_zeros = actual_results.replace('1',1).replace('9',0)
    proportion_error = sum(abs(predicted_zeros-actual_zeros))/len(y_test)

    matrix = confusion_matrix(predicted_activity, actual_results, labels = ['9','1'])
    f1 = f1_score(predicted_zeros, actual_zeros)

    return proportion_error, matrix, f1


# proportion_error, matrix, f1  = analyze_activitymodel(model, X_test, y_test)

def create_metrics_model(df):
    df_dummy = pd.get_dummies(df,columns=['activity'])
    X = df_dummy.drop(['elevation','distance','datetime','geometry'],axis=1)
    y = df_dummy[['distance','elevation']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    classifier = LinearRegression()
    model = classifier.fit(X_train, y_train)
    cross_val = cross_val_score(classifier, X, y, cv=3)
    return model, X_test, y_test, cross_val

# model_met, X_test_met, y_test_met, cross_val_met = create_metrics_model(encoded_data)

def get_error(model, X_test, y_test):
    predicted_results = pd.DataFrame(model.predict(X_test),columns=['distance','elevation'])
    actual_results = pd.DataFrame(y_test).reset_index(drop=True)
    error = abs(actual_results - predicted_results)/actual_results
    return actual_results, predicted_results, error

# actual_results_met, predicted_results_met, error_met = get_error(model_met, X_test_met, y_test_met)


# print(actual_results)
# print(predicted_results)
# print(error[(error['distance']>1) | (error['elevation']>1)])