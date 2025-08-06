import yfinance as yf
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, timedelta
from model_utils import scoreDataset
from reddit_utils import fetch_reddit_posts
from sentiment import analyze_sentiment
from sklearn.metrics import mean_absolute_error


def prepareRegressionData(dailySentiment):
    ticker = yf.Ticker("MSFT")
    tenYearYield = yf.Ticker("^TNX")
    stockData = ticker.history(period="5y")
    stockData["Target"] = stockData["Close"].shift(-1)
    stockData.dropna(inplace=True)
    stockData["PrevClose"] = stockData["Close"].shift(1)
    stockData["Return"] = (stockData["Close"] - stockData["PrevClose"]) / stockData["PrevClose"]
    stockData["MA5"] = stockData["Close"].rolling(5).mean()

    tenYearYieldData = tenYearYield.history(period="5y")
    tenYearYieldData = tenYearYieldData.reset_index()
    tenYearYieldData["Date"] = tenYearYieldData["Date"].dt.normalize()
    tenYearYieldData["Date"] = tenYearYieldData["Date"].dt.tz_localize(None)
    tenYearYieldData = tenYearYieldData[["Date", "Close"]]
    tenYearYieldData.columns = ["Date", "InterestRate"]

    stockData["PriceIncrease"] = (stockData['Close'].shift(-1) > stockData['Close']).astype(int)

    stockData = stockData.reset_index()
    stockData["Date"] = stockData["Date"].dt.normalize()
    stockData["Date"] = stockData["Date"].dt.tz_localize(None)

    stockData = pd.merge(stockData, tenYearYieldData, on="Date", how="left")
    stockData = pd.merge(stockData, dailySentiment, on="Date", how="left")
    stockData["RedditSentiment"] = stockData["RedditSentiment"].ffill().bfill()
    stockData["RedditSentiment"] = stockData["RedditSentiment"].shift(1)
    stockData['RollingSentiment_3d'] = stockData['RedditSentiment'].rolling(window=3).mean()


    stockData["RollingSentiment_1yr"] = stockData["RedditSentiment"].rolling(window=365).mean()
    stockData.ffill(inplace=True)

    features = ["Open", "High", "PrevClose", "MA5", "RollingSentiment_1yr", "InterestRate"]
    
    X = stockData[features]

    splitDate = (datetime.now() - timedelta(days=365)).date()
    train = stockData.loc[stockData["Date"] < pd.Timestamp(splitDate)]
    test = stockData.loc[stockData["Date"] >= pd.Timestamp(splitDate)]

    target = 'Target'

    X_train, y_train = train[features], train[target]
    X_val, y_val = test[features], test[target]

    Imputer = SimpleImputer(strategy="most_frequent")
    NImputer = SimpleImputer(strategy="mean")

    object_columns = X_train.select_dtypes(include="object").columns.tolist()
    numeric_columns = X_train.select_dtypes(exclude="object").columns.tolist()

    # Impute numeric columns always
    NImputed_X_Train = pd.DataFrame(NImputer.fit_transform(X_train[numeric_columns]), columns=numeric_columns, index=X_train.index)
    NImputed_X_val = pd.DataFrame(NImputer.transform(X_val[numeric_columns]), columns=numeric_columns, index=X_val.index)

    if len(object_columns) == 0:
        # No categorical columns: skip imputing or encoding categorical
        X_train_final = NImputed_X_Train
        X_val_final = NImputed_X_val
        OHEncoder = None
    else:
        # Impute categorical columns only if they exist
        OImputed_X_Train = pd.DataFrame(Imputer.fit_transform(X_train[object_columns].astype(str)), columns=object_columns, index=X_train.index)
        OImputed_X_val = pd.DataFrame(Imputer.transform(X_val[object_columns].astype(str)), columns=object_columns, index=X_val.index)

        # One-Hot Encode
        OHEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        OHtrain = pd.DataFrame(OHEncoder.fit_transform(OImputed_X_Train), columns=OHEncoder.get_feature_names_out(object_columns), index=X_train.index)
        OHval = pd.DataFrame(OHEncoder.transform(OImputed_X_val), columns=OHEncoder.get_feature_names_out(object_columns), index=X_val.index)

        # Combine numeric and encoded categorical
        X_train_final = pd.concat([NImputed_X_Train, OHtrain], axis=1)
        X_val_final = pd.concat([NImputed_X_val, OHval], axis=1)

    features_final = X_train_final.columns.tolist()
    scoreDataset(X_train=X_train_final, X_val=X_val_final, y_train=y_train, y_val=y_val, task="regression", stockData=stockData, features=features_final, OHEncoder=OHEncoder)

def prepareClassificationData(dailySentiment):
    ticker = yf.Ticker("MSFT")
    tenYearYield = yf.Ticker("^TNX")  # 10-year Treasury yield

    # Get historical price data (default daily)
    stockData = ticker.history(period="5y")  
    stockData.dropna(inplace=True)
    stockData["PrevClose"] = stockData["Close"].shift(1)
    stockData["Return"] = (stockData["Close"] - stockData["PrevClose"]) / stockData["PrevClose"]
    stockData["MA5"] = stockData["Close"].rolling(5).mean()
    stockData["Volatility5"] = stockData["Close"].rolling(5).std()

    
  
    # Create Momentum features
    stockData["Momentum_3d"] = stockData["Close"].pct_change(periods=3)
    stockData["Momentum_7d"] = stockData["Close"].pct_change(periods=7) 
    stockData["Momentum_14d"] = stockData["Close"].pct_change(periods=14)
    stockData["Momentum_30d"] = stockData["Close"].pct_change(periods=30)

    # Get 10-year Treasury yield data
    tenYearYieldData = tenYearYield.history(period="5y")
    tenYearYieldData = tenYearYieldData.reset_index()
    tenYearYieldData["Date"] = tenYearYieldData["Date"].dt.normalize()
    tenYearYieldData["Date"] = tenYearYieldData["Date"].dt.tz_localize(None)
    tenYearYieldData = tenYearYieldData[["Date", "Close"]]
    tenYearYieldData.columns = ["Date", "InterestRate"]



    #Create PriceDifference Column
    stockData["PriceIncrease"] = (stockData['Close'].shift(-1) > stockData['Close']).astype(int)

    #Merge with Reddit sentiment
    stockData = stockData.reset_index()  
    stockData["Date"] = stockData["Date"].dt.normalize()
    stockData["Date"] = stockData["Date"].dt.tz_localize(None)


    stockData = pd.merge(stockData, tenYearYieldData, on="Date", how="left") 

    stockData = pd.merge(stockData, dailySentiment, on="Date", how="left")
    stockData["RedditSentiment"] = stockData["RedditSentiment"].fillna(0)
    stockData["RollingSentiment_3d"] = stockData["RedditSentiment"].rolling(window=3).mean()
    stockData["RollingSentiment_7d"] = stockData["RedditSentiment"].rolling(window=7).mean()
    stockData["RollingSentiment_14d"] = stockData["RedditSentiment"].rolling(window=14).mean()
    stockData["RollingSentiment_30d"] = stockData["RedditSentiment"].rolling(window=30).mean()
    stockData["RollingSentiment_1yr"] = stockData["RedditSentiment"].rolling(window=365).mean()

    # Create Day of the Week Feature
    stockData["DayOfWeek"] = stockData["Date"].dt.day_name()

   
    

    features = ["Open", "High", "Volume", "PrevClose", "Return", "MA5", "Volatility5", "RollingSentiment_14d", "RollingSentiment_30d", "RollingSentiment_1yr", "InterestRate", "Momentum_3d", "Momentum_7d", "Momentum_14d", "Momentum_30d", "DayOfWeek"]

    X = stockData[features]

    #split training and test data
    splitDate = (datetime.now() - timedelta(days=365)).date()
    train = stockData.loc[stockData["Date"] < pd.Timestamp(splitDate)]
    test = stockData.loc[stockData["Date"] >= pd.Timestamp(splitDate)]

    target = 'PriceIncrease'


    X_train, y_train = train[features], train[target]
    X_val, y_val = test[features], test[target]

    Imputer = SimpleImputer(strategy="most_frequent")
    NImputer = SimpleImputer(strategy="mean")

    object_columns = X_train.select_dtypes(include="object").columns.tolist()
    numeric_columns = X_train.select_dtypes(exclude="object").columns.tolist()

    # Impute numeric columns always
    NImputed_X_Train = pd.DataFrame(NImputer.fit_transform(X_train[numeric_columns]), columns=numeric_columns, index=X_train.index)
    NImputed_X_val = pd.DataFrame(NImputer.transform(X_val[numeric_columns]), columns=numeric_columns, index=X_val.index)

    if len(object_columns) == 0:
        # No categorical columns: skip imputing or encoding categorical
        X_train_final = NImputed_X_Train
        X_val_final = NImputed_X_val
        OHEncoder = None
    else:
        # Impute categorical columns only if they exist
        OImputed_X_Train = pd.DataFrame(Imputer.fit_transform(X_train[object_columns].astype(str)), columns=object_columns, index=X_train.index)
        OImputed_X_val = pd.DataFrame(Imputer.transform(X_val[object_columns].astype(str)), columns=object_columns, index=X_val.index)

        # One-Hot Encode
        OHEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        OHtrain = pd.DataFrame(OHEncoder.fit_transform(OImputed_X_Train), columns=OHEncoder.get_feature_names_out(object_columns), index=X_train.index)
        OHval = pd.DataFrame(OHEncoder.transform(OImputed_X_val), columns=OHEncoder.get_feature_names_out(object_columns), index=X_val.index)

        # Combine numeric and encoded categorical
        X_train_final = pd.concat([NImputed_X_Train, OHtrain], axis=1)
        X_val_final = pd.concat([NImputed_X_val, OHval], axis=1)

    features_final = X_train_final.columns.tolist()

    scoreDataset(X_train=X_train_final, X_val=X_val_final, y_train=y_train, y_val=y_val, task="classification", stockData=stockData, features=features_final, OHEncoder=OHEncoder)
