from flask import Flask
import pandas as pd
import scraper
import wrangler
import forecast

app = Flask(__name__)

@app.route("/scrape/<string:ticker>")
def scrape(ticker:str):
    print("Scraping data...")
    return scraper.runFunc(ticker)

@app.route("/wrangle/<string:ticker>")
def wrangle(ticker:str):
    print("Wrangling data...")
    return wrangler.runFunc(ticker)

@app.route("/forecast/stats/<string:ticker>")
def forecast_stats(ticker:str):
    print("forecast stats...")
    return forecast.getStats(ticker)

@app.route("/forecast/get/<string:ticker>/<int:days>")
def forecast_doForecast(ticker:str,days:int):
    print("Performing forecast..")
    return forecast.doForecast(ticker,days)

@app.route("/forecast/hist/<string:ticker>")
def forecast_hist(ticker:str):
    print("forecast history data...")
    return forecast.getForecastHist(ticker)

@app.route("/forecast/byfile/<string:ticker>/<string:file>")
def forecast_byfile(ticker:str,file:str):
    print("Wrangling data...")
    return forecast.getForecastByFile(ticker,file)

@app.route("/forecast/latest/<string:ticker>")
def forecast_latest(ticker:str):
    print("Latest forecast data...")
    return forecast.getLatestForecast(ticker)

@app.route("/")
def home():
    return "Welcome!"


if __name__ == "__main__":
    app.run(debug=False)