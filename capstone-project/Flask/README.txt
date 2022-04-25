Machine Learning Engineering Bootcamp
Capstone Project
Tharaka Somaratna - UCSD ML bootcamp 2022 - 04 - 24

Project Files

Core files:
https://github.com/tharsoma/mec-mini-projects/tree/master/capstone-project/Production

Flask API files:
https://github.com/tharsoma/mec-mini-projects/tree/master/capstone-project/Flask

Introduction

This capstone project is about using what people talk on stocktwit.com about stocks, and using it to predict the stock market. 

The core data used on this project are obtained from both the stocktwits.com and yahoofinances.com. To get user data, stocktwit api was used. Here is an example of a person mentioning a stock. 

Process

Here is a list of actions performed to obtain our final result.

Scrape stocktwit data from its API based on the stock ticker
Get their set sentiment, and also derive their sentiment based on their message by using bag of words method.
Scale sentiments from [0 to 1]
add stocks prices to corresponding days based on  the stock ticker
Once the data is cleaned, use linear regression to predict stock values for the given number of days.

API Calls

http://127.0.0.1:5000/scrape/<string:ticker>
This will scrape 6 months worth of stocktwit data of a given stock ticker.

http://127.0.0.1:5000/wrangle/<string:ticker>
This will clean up all the data and make it ready for ML

http://127.0.0.1:5000/forecast/stats/<string:ticker>
This will obtain Linear regression model’s Model Coefficients, Mean Absolute Error, Coefficient of Determination

http://127.0.0.1:5000/forecast/get/<string:ticker>/<int:days>
This will generate the forecast for the given ticker and the number of days into the future.

http://127.0.0.1:5000/forecast/hist/<string:ticker>
This will return all the previous forecasts that was generated

http://127.0.0.1:5000/forecast/byfile/<string:ticker>/<string:file>
This will return a forecast by its file name.

http://127.0.0.1:5000/forecast/latest/<string:ticker>
This will the most recent forecast that was generated.
