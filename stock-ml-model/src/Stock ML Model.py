import os
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import yfinance as yf
from datetime import datetime, timedelta
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path("C:\\Users\\wyatt\\Downloads\\Progamming projets\\stock-ml-model\\Api.env"))
#Load redditAPI
reddit = praw.Reddit(client_id=os.getenv("Reddit_client_id"),
                     client_secret=os.getenv("Reddit_client_secret"),
                    user_agent=os.getenv("Reddit_user_agent"))
# Function to fetch Reddit posts
def fetchRedditPosts(subreddits, keyword, limit=100):
    postList = []
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        posts = subreddit.search(keyword, time_filter='year', limit=1000)
    
        for submission in posts:
            post = {
                'title': submission.title,
                'created_utc': submission.created_utc,
                'selftext': submission.selftext
            }
            postList.append(post)
    postdf = pd.DataFrame(postList)
    print(len(postdf), "posts fetched from Reddit")
    postdf = postdf.rename(columns={'created_utc': 'Date'})
    postdf['Date'] = pd.to_datetime(postdf['Date'], unit='s')
    postdf['Date'] = postdf['Date'].dt.normalize()
    postdf.sort_values(by="Date", ascending=False,inplace=True)
    # Convert to DataFrame
    return postdf

investingSubreddits = ["investing", "stocks", "wallstreetbets", "microsoft"]
postdf = fetchRedditPosts(investingSubreddits, "microsoft")

# Function to fetch Reddit sentiment
analyzer = SentimentIntensityAnalyzer()
postdf["sentiment_score"] = postdf.apply(lambda row: analyzer.polarity_scores((row["title"] or "") + " " + (row["selftext"] or ""))["compound"], axis=1)
dailySentiment = postdf.groupby("Date")["sentiment_score"].mean().reset_index()   
dailySentiment.columns = ["Date", "RedditSentiment"]

def scoreDataset(X_train, X_val, y_train, y_val, nEstimators, max_depth, min_samples_leaf, min_samples_split, task, stockData, features):
    if task == "regression":
        stockModel = RandomForestRegressor(random_state=1,
                                    max_depth=max_depth,
                                    n_estimators=nEstimators
                                        , min_samples_leaf=min_samples_leaf,
                                        min_samples_split=min_samples_split
                                    )
        stockModel.fit(X_train,y_train)
        stockPredictions = stockModel.predict(X_val)
        stockMae = mean_absolute_error(y_val, stockPredictions)
        maeAverage = 100 * (stockMae/y_val.mean())
        print("Validation MAE for Random Forest Model: {:,.2f}".format(maeAverage), "%")
        latest_data = stockData.iloc[-1]
        tomorrow_features = pd.DataFrame([latest_data[features]])

        # Predict tomorrow's closing price
        tomorrow_pred = stockModel.predict(tomorrow_features)
        print(f"Predicted closing price for tomorrow: ${tomorrow_pred[0]:.2f}")
        findFeatureImportance(stockModel, features)
    elif task == "classification":
        stockModel = RandomForestClassifier(random_state=1,
                                max_depth=max_depth,
                                n_estimators=nEstimators
                                    , min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split
                                )
        stockModel.fit(X_train,y_train)
        stockPredictions = stockModel.predict(X_val)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        print("Accuracy: {:.2f}".format(accuracy_score(y_val, stockPredictions)))
        print("F1 Score: {:.2f}".format(f1_score(y_val, stockPredictions)))
        print("Precision: {:.2f}".format(precision_score(y_val, stockPredictions)))
        print("Recall: {:.2f}".format(recall_score(y_val, stockPredictions)))
        # Predict wheter the stock price will increase tomorrow
        latest_data = stockData.iloc[-1]
        tomorrow_features = pd.DataFrame([latest_data[features]])
        tomorrow_pred = stockModel.predict(tomorrow_features)
        if tomorrow_pred[0] == 1:
            print("Predicted stock price will increase tomorrow.")
        else:
            print("Predicted stock price will decrease tomorrow.")
        findFeatureImportance(stockModel, features)




def findFeatureImportance(stockModel, features):
    # After training
    importances = stockModel.feature_importances_
    # Pair each importance with the feature name
    feature_importance = pd.Series(importances, index=features)
    # Sort descending
    feature_importance = feature_importance.sort_values(ascending=False)
    # Print
    print(feature_importance)
def prepareRegressionData():
    ticker = yf.Ticker("MSFT")

    # Get historical price data (default daily)
    stockData = ticker.history(period="1y")  # options: '1d', '5d', '1mo', '1y', '5y', 'max'
    stockData["Target"] = stockData["Close"].shift(-1)
    stockData.dropna(inplace=True)
    stockData["PrevClose"] = stockData["Close"].shift(1)
    stockData["Return"] = (stockData["Close"] - stockData["PrevClose"]) / stockData["PrevClose"]
    stockData["MA5"] = stockData["Close"].rolling(5).mean()
    stockData["Volatility5"] = stockData["Close"].rolling(5).std()

    #Create PriceDifference Column
    stockData["PriceIncrease"] = (stockData['Close'].shift(-1) > stockData['Close']).astype(int)

    #Merge with Reddit sentiment
    stockData = stockData.reset_index()  # Make 'Date' a column
    stockData["Date"] = stockData["Date"].dt.normalize()
    stockData["Date"] = stockData["Date"].dt.tz_localize(None) 

    stockData = pd.merge(stockData, dailySentiment, on="Date", how="left")
    stockData["RedditSentiment"] = stockData["RedditSentiment"].fillna(0)


    features = ["RedditSentiment","Open", "High", "Volume", "PrevClose", "Return", "MA5", "Volatility5"]

    X = stockData[features]

    #split training and test data
    splitDate = (datetime.now() - timedelta(days=45)).date()
    train = stockData.loc[stockData["Date"] < pd.Timestamp(splitDate)]
    test = stockData.loc[stockData["Date"] >= pd.Timestamp(splitDate)]

    target = 'Target'


    X_train, y_train = train[features], train[target]
    X_val, y_val = test[features], test[target]
    #Create Imputed Data
    Imputer = SimpleImputer()
    Imputed_X_Train = pd.DataFrame(Imputer.fit_transform(X_train))
    Imputed_X_val = pd.DataFrame(Imputer.transform(X_val))

    #Re-add Columns

    Imputed_X_Train.columns = X_train.columns
    Imputed_X_val.columns = X_val.columns
    scoreDataset(Imputed_X_Train, Imputed_X_val, y_train, y_val, nEstimators=100, max_depth=20, min_samples_leaf=1, min_samples_split=2, task="regression", stockData=stockData, features=features)



def prepareClassificationData():
    ticker = yf.Ticker("MSFT")

    # Get historical price data (default daily)
    stockData = ticker.history(period="1y")  # options: '1d', '5d', '1mo', '1y', '5y', 'max'
    stockData.dropna(inplace=True)
    stockData["PrevClose"] = stockData["Close"].shift(1)
    stockData["Return"] = (stockData["Close"] - stockData["PrevClose"]) / stockData["PrevClose"]
    stockData["MA5"] = stockData["Close"].rolling(5).mean()
    stockData["Volatility5"] = stockData["Close"].rolling(5).std()

    #Create PriceDifference Column
    stockData["PriceIncrease"] = (stockData['Close'].shift(-1) > stockData['Close']).astype(int)

    #Merge with Reddit sentiment
    stockData = stockData.reset_index()  # Make 'Date' a column
    stockData["Date"] = stockData["Date"].dt.normalize()
    stockData["Date"] = stockData["Date"].dt.tz_localize(None) 

    stockData = pd.merge(stockData, dailySentiment, on="Date", how="left")
    stockData["RedditSentiment"] = stockData["RedditSentiment"].fillna(0)


    features = ["RedditSentiment","Open", "High", "Volume", "PrevClose", "Return", "MA5", "Volatility5"]

    X = stockData[features]

    #split training and test data
    splitDate = (datetime.now() - timedelta(days=45)).date()
    train = stockData.loc[stockData["Date"] < pd.Timestamp(splitDate)]
    test = stockData.loc[stockData["Date"] >= pd.Timestamp(splitDate)]

    target = 'PriceIncrease'


    X_train, y_train = train[features], train[target]
    X_val, y_val = test[features], test[target]
    #Create Imputed Data
    Imputer = SimpleImputer()
    Imputed_X_Train = pd.DataFrame(Imputer.fit_transform(X_train))
    Imputed_X_val = pd.DataFrame(Imputer.transform(X_val))

    #Re-add Columns

    Imputed_X_Train.columns = X_train.columns
    Imputed_X_val.columns = X_val.columns
    scoreDataset(X_train=Imputed_X_Train, X_val=Imputed_X_val, y_train=y_train, y_val=y_val, nEstimators=100, max_depth=20, min_samples_leaf=1, min_samples_split=2, task="classification", stockData=stockData, features=features)
prepareClassificationData()
#for i in features:
    #correlation = stockData[i].corr(stockData["Target"])
    #print(f"Correlation between {i} and stock price: {correlation:.2f}")




