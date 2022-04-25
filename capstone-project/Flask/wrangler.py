import os, json
import pandas as pd
from datetime import timedelta
import yfinance as yf
from sklearn import linear_model 

#stock = yf.Ticker("AAPL")
#hist = stock.history(period="max")

def subtract_days_from_date(date, days):
    """Subtract days from a date and return the date.
    
    Args: 
        date (string): Date string in YYYY-MM-DD format. 
        days (int): Number of days to subtract from date
    
    Returns: 
        date (date): Date in YYYY-MM-DD with X days subtracted. 
    """
    
    subtracted_date = pd.to_datetime(date) - timedelta(days=days)
    subtracted_date = subtracted_date.strftime("%Y-%m-%d")

    return subtracted_date

def append_all_json_data(folderPath: str, list_of_jsons) -> pd.DataFrame:
    df = pd.DataFrame()
    for element in list_of_jsons:
        full_path = folderPath+"/"+element
        #print (full_path)
        df = df.append(pd.read_json(full_path))
    return df

def get_raw_sentiment(sentimentPath):
    if sentimentPath == None:
        return None
    else:
        return sentimentPath["basic"]
    
def get_stockprice_at_time(d,hist):
    #stock = yf.Ticker(ticker)
    #hist = stock.history(period="max")
    day = pd.to_datetime(d)
    #print(day.day)
    _day = str(day.year)+"-"+str(day.month)+"-"+str(day.day)
    try:
        val = hist.loc[_day].Close
    except KeyError:
        d = subtract_days_from_date(_day, 1)
        val = get_stockprice_at_time(d,hist)
        
    return round(val,3)

def get_openprice_at_time(d,hist):
    #stock = yf.Ticker(ticker)
    #hist = stock.history(period="max")
    day = pd.to_datetime(d)
    #print(day.day)
    _day = str(day.year)+"-"+str(day.month)+"-"+str(day.day)
    try:
        val = hist.loc[_day].Open
    except KeyError:
        d = subtract_days_from_date(_day, 1)
        val = get_openprice_at_time(d,hist)
        
    return round(val,3)

def get_volume_at_time(d,hist):
    #stock = yf.Ticker(ticker)
    #hist = stock.history(period="max")
    day = pd.to_datetime(d)
    #print(day.day)
    _day = str(day.year)+"-"+str(day.month)+"-"+str(day.day)
    try:
        val = hist.loc[_day].Volume
    except KeyError:
        d = subtract_days_from_date(_day, 1)
        val = get_volume_at_time(d,hist)
        
    return round(val,3)

#Bag of words for sentiment prediction
#https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv('bagofwords_data.csv')
dataset= dataset[dataset["Sentiment"].str.contains("neutral")==False]
dataset=dataset.reset_index(drop=True)
sentiment = {'positive': 1,'negative': 0}
dataset.Sentiment = [sentiment[item] for item in dataset.Sentiment]
corpus = []


    
for i in range(0,2712):
    review = re.sub('[^a-zA-Z]',' ',dataset['Sentence'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
#train, test = train_test_split(dataset, test_size = 0.33, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.79, random_state=42)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

def get_bow_sentiment(msg):
    msg_review = msg
    msg_review = re.sub('[^a-zA-Z]',' ',msg_review)
    msg_review = msg_review.lower()
    msg_review = msg_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    msg_review = [ps.stem(word) for word in msg_review if not word in set(all_stopwords)]
    msg_review = ' '.join(msg_review)
    user_corpus = [msg_review]
    new_x_test = cv.transform(user_corpus).toarray()
    new_y_pred = classifier.predict(new_x_test)
    if new_y_pred[0] == 0:
        return 0#'Bearish'
    else:
        return 1#'Bullish'

#from sklearn.metrics import confusion_matrix, accuracy_score
#cm = confusion_matrix(y_test,y_pred)
#print(cm)
#accuracy_score(y_test,y_pred)
def categoryize_sentiment(sentiment):
    if sentiment == "Bullish":
        return 1
    if sentiment == "Bearish":
        return 0
    if sentiment == None or sentiment == "None":
        return 0.5

def avg_sentiment(s1,s2):
    _s1 = s1
    _s2 = s2
    if s1 == None:
        _s1 = 0.5
    if s2 == None:
        _s2 = 0.5
    return (_s1+_s2)/2


def runFunc(ticker:str):
    
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max")
    if not os.path.isdir("stocks/"+ticker+"/Wrangled"):
        os.mkdir("stocks/"+ticker+"/Wrangled")
    file_path = "stocks/"+ticker+"/Wrangled/"+ticker+"_wrangled_data.csv"
    if os.path.exists(file_path):
        os.remove(file_path)
    
    path_to_json = 'stocks/'+ticker+'/Raw/'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    raw_json_data = append_all_json_data(path_to_json,json_files)
    raw_json_data=raw_json_data.reset_index(drop=True)
    
    new_dataframe = pd.DataFrame(columns=["time_created","username","message","old_sentiment","derived_sentiment","sentiment","open","close","volume"])
    for i in range(raw_json_data.shape[0]):
        ts = raw_json_data["created_at"][i]
        day = str(ts.year)+"-"+str(ts.month)+"-"+str(ts.day)
        new_dataframe.loc[i]= [raw_json_data["created_at"][i],raw_json_data["user"][i]["username"],raw_json_data["body"][i],categoryize_sentiment(get_raw_sentiment(raw_json_data["entities"][i]["sentiment"])),get_bow_sentiment(raw_json_data["body"][i]),avg_sentiment(categoryize_sentiment(get_raw_sentiment(raw_json_data["entities"][i]["sentiment"])),get_bow_sentiment(raw_json_data["body"][i])),get_openprice_at_time(day,hist),get_stockprice_at_time(day,hist),get_volume_at_time(day,hist)]
        
    ndf = new_dataframe.groupby(by=new_dataframe['time_created'].dt.date).mean()
    X = ndf[['sentiment', 'open']]
    y = ndf['close'] 
    regr = linear_model.LinearRegression()
    regr.fit(X, y) 
    ndf['predicted_closing'] = ndf.apply(lambda x: regr.predict([[x['sentiment'],x['open']]])[0], axis=1)
    ndf.to_csv("stocks/"+ticker+"/Wrangled/"+ticker+"_wrangled_data.csv")
    
    y = {"wrangle success!":ndf.to_json()}
    return (json.dumps(y))