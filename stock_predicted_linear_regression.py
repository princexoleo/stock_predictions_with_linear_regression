"""##Import Libraries"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn import linear_model
import tkinter
import matplotlib
matplotlib.use('TkAgg')

"""##Data Preprocessing"""

def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out);#creating new column called label with the last 5 rows are nan
    X = np.array(df[[forecast_col]]); #creating the feature array
    X = preprocessing.scale(X) #processing the feature array
    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method
    X = X[:-forecast_out] # X that will contain the training and testing
    label.dropna(inplace=True); #dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size) #cross validation 

    response = [X_train,X_test , Y_train, Y_test , X_lately];
    return response;


def linear_regression_fun(df, name):
    """##Ploting"""

    # df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
    # df.index = df['Date']
    # plt.figure(figsize=(16,8))
    # plt.plot(df['Close'], label='Close Price history')

    """##Split Datasets"""

    forecast_col = 'Close'#choosing which column to forecast
    forecast_out = 5 #how far to forecast 
    test_size = 0.2; #the size of my test set
    X_train, X_test, Y_train, Y_test , X_lately =prepare_data(df,forecast_col,forecast_out,test_size)
    print(X_lately) ## This is our predicting test data

    """##Use Linear Regression Model"""

    learner = linear_model.LinearRegression()
    learner.fit(X_train,Y_train); #training the linear regression model
    score=learner.score(X_test,Y_test);#testing the linear regression model
    forecast= learner.predict(X_lately); #set that will contain the forecasted data
    print(score)
    print(forecast)

    """##Predicted Value Score Plot"""

    # plt.figure(figsize=forecast.shape)
    # plt.plot(forecast, label='Predicted price')
    plt.title(name)
    plt.plot(X_lately,forecast, label ="predicted close price")
    #plt.plot(X_lately,label="Previous day close price")
    plt.ylabel("Predicted Price")
    plt.xlabel("Previous Day Close Price")
    plt.show()


"""##Data Load"""

for i in range(3):
    if i==0:
        df = pd.read_excel(r'Apple.xlsx')
        print(df.head())
        linear_regression_fun(df, "Apple");
    elif i == 1:
        df = pd.read_excel(r'Google.xlsx')
        print(df.head())
        linear_regression_fun(df, "Google")
    else:
        df = pd.read_excel(r'Microsoft.xlsx')
        print(df.head())
        linear_regression_fun(df, "Microsoft")



