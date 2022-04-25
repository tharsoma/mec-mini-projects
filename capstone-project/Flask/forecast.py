import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import json
import os

#get latest forect -> returns latest
#perform forecast -> Ticker + Days
#get forecast history ->returns all file names in a json
#get forecast by date -> Ticker + date

def getStats(ticker:str):
    
    file = "stocks/"+ticker+"/Wrangled/"+ticker+"_wrangled_data.csv"
    if os.path.exists(file) == False:
        y = {"file not found":0}
        return (json.dumps(y))


    df = pd.read_csv(file)    
    X_train, X_test, y_train, y_test = train_test_split(df[['close']], df[['predicted_closing']], test_size=.2)
    model = LinearRegression()
    # Train the model
    model.fit(X_train, y_train)
    # Use model to make predictions
    y_pred = model.predict(X_test)
    x = {
      "Model Coefficients": model.coef_[0][0],
      "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
      "Coefficient of Determination": r2_score(y_test, y_pred)
    }    
    return (json.dumps(x))

def doForecast(ticker:str,days:int):
    if not os.path.isdir("stocks/"+ticker+"/Forecast"):
        os.mkdir("stocks/"+ticker+"/Forecast")
    file = "stocks/"+ticker+"/Wrangled/"+ticker+"_wrangled_data.csv"
    if os.path.exists(file) == False:
        y = {"file not found":0}
        return (json.dumps(y))

    df = pd.read_csv(file)    
    X_train, X_test, y_train, y_test = train_test_split(df[['close']], df[['predicted_closing']], test_size=.2)
    model = LinearRegression()
    # Train the model
    model.fit(X_train, y_train)
    # Use model to make predictions
    y_pred = model.predict(X_test)
    x_forecast = np.array(df.drop(['predicted_closing','volume','open','time_created','sentiment','old_sentiment','derived_sentiment'],1))[-days:]
    lr_prediction = model.predict(x_forecast)
    new_dataframe = pd.DataFrame(columns=["day - close"])
    first_day = pd.to_datetime(df.time_created.tail(1).values[0]) + timedelta(days=1)
    for i in range(lr_prediction.shape[0]):       
        new_dataframe.loc[i]= [first_day.strftime("%Y-%m-%d")+":"+str(round(lr_prediction[i][0],3))]
        first_day=first_day+timedelta(days=1)
    new_dataframe.to_csv("stocks/"+ticker+"/Forecast/"+ticker+"_forecast_"+datetime.today().strftime("%Y-%m-%d-%HH-%MM-%SS")+".csv")
    return (new_dataframe.to_json())

def getForecastHist(ticker:str):
    csv_files = [pos_json for pos_json in os.listdir("stocks/"+ticker+"/Forecast/") if pos_json.endswith('.csv')]
    y = {"forecast history":csv_files}
    return (json.dumps(y))
    
def getForecastByFile(ticker,fileName:str):
    df = pd.read_csv("stocks/"+ticker+"/Forecast/"+fileName)
    df=df.drop(columns="Unnamed: 0")
    return df.to_json()
    
def getLatestForecast(ticker:str):
    csv_files = [pos_json for pos_json in os.listdir("stocks/"+ticker+"/Forecast/") if pos_json.endswith('.csv')]
    df = pd.read_csv("stocks/"+ticker+"/Forecast/"+csv_files[-1])
    df=df.drop(columns="Unnamed: 0")
    return df.to_json()