import os
import pandas as pd
from scipy.stats import randint
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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import wbdata


CACHE_FILE = "reddit_posts_cache.csv"

load_dotenv(dotenv_path="c:/Users/wyatt/Downloads/Progamming projets/stock-ml-model/src/Api.env")

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Function to fetch Reddit posts
def fetchRedditPosts(subreddit, keyword, timeframe, sortvalue):
    expected_columns = ["title", "selftext", "Date"]
    posts = []
    try:
        subreddit = reddit.subreddit(subreddit)
        for submission in subreddit.search(query=keyword, sort=sortvalue, time_filter=timeframe, limit=1000):
            created_time = datetime.fromtimestamp(submission.created_utc)
            if "microsoft" in submission.title.lower() or "msft" in submission.title.lower():
                posts.append({
                    "title": submission.title,
                    "Date": created_time
            })
    except Exception as e:
        print(f"Failed to fetch from r/{subreddit}: {e}")
        return pd.DataFrame(columns=expected_columns)

    postdf = pd.DataFrame(posts)
    if postdf.empty:
        print(f"0 posts fetched from r/{subreddit}")
        return pd.DataFrame(columns=expected_columns)
    
    postdf["Date"] = pd.to_datetime(postdf["Date"]).dt.normalize()
    print(f"{len(postdf)} posts fetched from r/{subreddit}")
    return postdf

# Function to fetch Reddit sentiment


def findCorrelation(stockData, features, cor_target):
    
    for col in features:
            if stockData[col].equals(stockData[cor_target]):
                print(f"WARNING: Feature {col} is identical to Target!")
    subset = stockData[features + [cor_target]].dropna()
    for i in features:
        correlation = subset[i].corr(stockData[cor_target])
        print(f"Correlation between {i} and stock price: {correlation:.2f}")

def scoreDataset(X_train, X_val, y_train, y_val, task, stockData, features):
    if task == "regression":
    
        XGBmodel = XGBRegressor(
            random_state=1,
            max_depth=5,
            n_estimators=100,
            colsample_bytree=0.7,
            learning_rate=0.1
)

        print("Training XGBoost Model...")
        XGBmodel.fit(X_train, y_train)
        stockPredictions = XGBmodel.predict(X_val)
        stockMae = mean_absolute_error(y_val, stockPredictions)
        maeAverage = 100 * (stockMae/y_val.mean())
        print("Validation MAE for XGBoost Model: {:,.2f}".format(maeAverage), "%")
        latest_data = stockData.iloc[-1]
        tomorrow_features = pd.DataFrame([latest_data[features]])

        # Predict tomorrow's closing price
        tomorrow_pred = XGBmodel.predict(tomorrow_features)
        print(f"Predicted closing price for tomorrow: ${tomorrow_pred[0]:.2f}")
        findFeatureImportance(XGBmodel, features)
        findCorrelation(stockData, features, "Target")

    elif task == "classification":
# Random Search for hyperparameter tuning
        XGBmodel =  XGBClassifier(
            subsample=0.7,
            reg_lambda=1,
            reg_alpha=0.1,
            n_estimators=400,
            min_child_weight=7,
            max_depth=4,
            learning_rate=0.2,
            gamma=0,
            colsample_bytree=0.6,
            eval_metric="logloss",
            random_state=1
        )
#         param_dist = {
#             'n_estimators': [100, 200, 300, 400],
#             'learning_rate': [0.01, 0.05, 0.1, 0.2],
#             'max_depth': [3, 4, 5, 6, 8],
#             'subsample': [0.6, 0.7, 0.8, 1.0],
#             'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
#             'gamma': [0, 1, 5],
#             'min_child_weight': [1, 3, 5, 7],
#             'reg_alpha': [0, 0.1, 1],
#             'reg_lambda': [1, 1.5, 2],
# }
#         randomSearch = RandomizedSearchCV(
#             XGBmodel,
#             param_distributions=param_dist,
#             n_iter=30,
#             scoring='accuracy',
#             cv=3,
#             verbose=2,
#             n_jobs=-1)
#         randomSearch.fit(X_train, y_train)
#         print("Best parameters found: ", randomSearch.best_params_)
#         XGBmodel = randomSearch.best_estimator_

        XGBmodel.fit(X_train,y_train)
        stockPredictions = XGBmodel.predict(X_val)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        print("Accuracy: {:.2f}".format(accuracy_score(y_val, stockPredictions)))
        print("F1 Score: {:.2f}".format(f1_score(y_val, stockPredictions)))
        print("Precision: {:.2f}".format(precision_score(y_val, stockPredictions)))
        print("Recall: {:.2f}".format(recall_score(y_val, stockPredictions)))
        # Predict wheter the stock price will increase tomorrow
        latest_data = stockData.iloc[-1]
        tomorrow_features = pd.DataFrame([latest_data[features]])
        tomorrow_pred = XGBmodel.predict(tomorrow_features)
        if tomorrow_pred[0] == 1:
            print("Predicted stock price will increase tomorrow.")
        else:
            print("Predicted stock price will decrease tomorrow.")
        findFeatureImportance(XGBmodel, features)
        findCorrelation(stockData, features, "PriceIncrease")




def findFeatureImportance(stockModel, features):
    # After training
    importances = stockModel.feature_importances_
    # Pair each importance with the feature name
    feature_importance = pd.Series(importances, index=features)
    # Sort descending
    feature_importance = feature_importance.sort_values(ascending=False)
    # Print
    print("Feature Importances:")
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.4f}")

def prepareRegressionData():
    ticker = yf.Ticker("MSFT")
    tenYearYield = yf.Ticker("^TNX")
    # Get historical price data (default daily)
    stockData = ticker.history(period="5y")  # options: '1d', '5d', '1mo', '1y', '5y', 'max'
    stockData["Target"] = stockData["Close"].shift(-1)
    stockData.dropna(inplace=True)
    stockData["PrevClose"] = stockData["Close"].shift(1)
    stockData["Return"] = (stockData["Close"] - stockData["PrevClose"]) / stockData["PrevClose"]
    stockData["MA5"] = stockData["Close"].rolling(5).mean()
    stockData["Volatility5"] = stockData["Close"].rolling(5).std()
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
    stockData = stockData.reset_index()  # Make 'Date' a column
    stockData["Date"] = stockData["Date"].dt.normalize()
    stockData["Date"] = stockData["Date"].dt.tz_localize(None) 

    stockData = pd.merge(stockData, tenYearYieldData, on="Date", how="left") 

    stockData = pd.merge(stockData, dailySentiment, on="Date", how="left")
    missingReddit = stockData["RedditSentiment"].isnull().sum()
    print(f"Missing sentiment scores: {missingReddit}")
    stockData["RedditSentiment"] = stockData["RedditSentiment"].ffill().bfill()
    stockData["RedditSentiment"] = stockData["RedditSentiment"].shift(1)  # Shift to align with the next day's price
    stockData['RollingSentiment_3d'] = stockData['RedditSentiment'].rolling(window=3).mean()

    #Add Rolling Sentiment Features
    stockData["RollingSentiment_1yr"] = stockData["RedditSentiment"].rolling(window=365).mean()
    stockData.ffill(inplace=True)
    features = ["Open", "High", "PrevClose",  "MA5",  "RollingSentiment_1yr", "InterestRate"]

    X = stockData[features]

    #split training and test data
    splitDate = (datetime.now() - timedelta(days=365)).date()
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
    tenYearYield = yf.Ticker("^TNX")  # 10-year Treasury yield

    # Get historical price data (default daily)
    stockData = ticker.history(period="5y")  
    stockData.dropna(inplace=True)
    stockData["PrevClose"] = stockData["Close"].shift(1)
    stockData["Return"] = (stockData["Close"] - stockData["PrevClose"]) / stockData["PrevClose"]
    stockData["MA5"] = stockData["Close"].rolling(5).mean()
    stockData["Volatility5"] = stockData["Close"].rolling(5).std()

    # Get 10-year Treasury yield data
    tenYearYieldData = tenYearYield.history(period="5y")
    tenYearYieldData = tenYearYieldData.reset_index()
    tenYearYieldData["Date"] = tenYearYieldData["Date"].dt.normalize()
    tenYearYieldData["Date"] = tenYearYieldData["Date"].dt.tz_localize(None)
    tenYearYieldData = tenYearYieldData[["Date", "Close"]]
    tenYearYieldData.columns = ["Date", "InterestRate"]

    #Get World Bank Data
    startTime = (datetime.now() - timedelta(days=365*5)).date()
    endTime = datetime.now().date()

    InflationData = wbdata.get_dataframe(
        {"FP.CPI.TOTL.ZG": "InflationRate"},
        country="US",
        date=(startTime, endTime),
        convert_date=True
    ).reset_index()
    print(f"Inflation data shape: {InflationData.shape}")

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
    

    features = ["Open", "High", "Volume", "PrevClose", "Return", "MA5", "Volatility5", "RollingSentiment_14d", "RollingSentiment_30d", "RollingSentiment_1yr", "InterestRate"]

    X = stockData[features]

    #split training and test data
    splitDate = (datetime.now() - timedelta(days=365)).date()
    train = stockData.loc[stockData["Date"] < pd.Timestamp(splitDate)]
    test = stockData.loc[stockData["Date"] >= pd.Timestamp(splitDate)]

    target = 'PriceIncrease'


    X_train, y_train = train[features], train[target]
    X_val, y_val = test[features], test[target]
   


    print(y_train.value_counts(normalize=True))
    scoreDataset(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, task="classification", stockData=stockData, features=features)


def main():
    if os.path.exists(CACHE_FILE):
        print("Loading cached Reddit posts...")
        combined_posts = pd.read_csv(CACHE_FILE, parse_dates=["Date"])
    else:
        investingSubreddits = ["investing", "stocks", "wallstreetbets",
        "StockMarket", "financialindependence", "dividends",
        "pennystocks", "options", "algotrading"]
        timeframes = [ "year", "all"]
        sortvalues = [ "top", "hot"]
        allPosts = []
        for i in range(len(investingSubreddits)):
            for j in range(len(timeframes)):
                for k in range(len(sortvalues)):
                    postdf = fetchRedditPosts(investingSubreddits[i], "microsoft", timeframe=timeframes[j], sortvalue=sortvalues[k])
                    allPosts.append(postdf)
                    time.sleep(1)
        combined_posts = pd.concat(allPosts).drop_duplicates().reset_index(drop=True)
        combined_posts.to_csv(CACHE_FILE, index=False)

    # Analyze sentiment
    try:
        allPosts
        combined_posts = pd.concat(allPosts).drop_duplicates().reset_index(drop=True)
    except Exception as e:
        x=1

    if combined_posts.empty or "Date" not in combined_posts.columns:
        print("No Reddit data collected. Exiting.")
        return
    analyzer = SentimentIntensityAnalyzer()
    combined_posts["sentiment_score"] = combined_posts.apply(
    lambda row: analyzer.polarity_scores(
        (row["title"] if pd.notnull(row["title"]) else "") + " " +
        (row["selftext"] if pd.notnull(row["selftext"]) else "")
    )["compound"], axis=1
)
        
    global dailySentiment
    dailySentiment = combined_posts.groupby("Date")["sentiment_score"].mean().reset_index()
    dailySentiment.columns = ["Date", "RedditSentiment"]

    
 
    # Choose which preparation function to run:
    prepareClassificationData()
    #prepareRegressionData()

if __name__ == "__main__":
    main()

