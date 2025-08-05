import os
import pandas as pd
from scipy.stats import randint
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
from sklearn.preprocessing import OneHotEncoder


CACHE_FILE = "reddit_posts_cache.csv"
#os.remove(CACHE_FILE) 

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
            if ("microsoft" in submission.title.lower() or "msft" in submission.title.lower()) and created_time.date() >= datetime.now().date() - timedelta(days=365*5):
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

def findFeatureImportance(stockModel, features):
    # After training
    importances = stockModel.feature_importances_
    feature_importance = pd.Series(importances, index=features)
    feature_importance = feature_importance.sort_values(ascending=False)
    print("Feature Importances:")
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.4f}")

def scoreDataset(X_train, X_val, y_train, y_val, task, stockData, features, OHEncoder):
    if task == "regression":
    
        XGBmodel = XGBRegressor(
            random_state=1,
            max_depth=8,
            gamma=5,
            n_estimators=100,
            colsample_bytree=0.7,
            learning_rate=0.1,
            subsample=0.6,
            reg_lambda=2,
            reg_alpha=1,
            min_child_weight=5,
)
#         param_dist = {
#              'n_estimators': [100, 200, 300, 400],
#              'learning_rate': [0.01, 0.05, 0.1, 0.2],
#              'max_depth': [3, 4, 5, 6, 8],
#              'subsample': [0.6, 0.7, 0.8, 1.0],
#              'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
#             'gamma': [0, 1, 5],
#              'min_child_weight': [1, 3, 5, 7],
#              'reg_alpha': [0, 0.1, 1],
#              'reg_lambda': [1, 1.5, 2],
#  }
#         randomSearch = RandomizedSearchCV(
#              XGBmodel,
#              param_distributions=param_dist,
#              n_iter=30,
#              scoring="neg_mean_absolute_error",
#              cv=3,
#              verbose=2,
#              n_jobs=-1)
#         randomSearch.fit(X_train, y_train)
#         print("Best parameters found: ", randomSearch.best_params_)
#         XGBmodel = randomSearch.best_estimator_

        print("Training XGBoost Model...")
        XGBmodel.fit(X_train, y_train)
        stockPredictions = XGBmodel.predict(X_val)
        stockMae = mean_absolute_error(y_val, stockPredictions)
        maeAverage = 100 * (stockMae/y_val.mean())
        print("Validation MAE for XGBoost Model: {:,.2f}".format(maeAverage), "%")
        latest_data = stockData.iloc[-1]
        object_columns = X_train.select_dtypes(include="object").columns.tolist()
        features = X_train.columns.tolist() 
        
        tomorrow_features = encode_new_row(latest_data, OHEncoder, object_columns, features)
        tomorrow_features = tomorrow_features.astype(float) 

        # Predict tomorrow's closing price
        tomorrow_pred = XGBmodel.predict(tomorrow_features)
        print(f"Predicted closing price for tomorrow: ${tomorrow_pred[0]:.2f}")
        findFeatureImportance(XGBmodel, features)
        df_for_corr = X_train.copy()
        df_for_corr["Target"] = y_train
        findCorrelation(df_for_corr, features, "Target")

    elif task == "classification":
# Random Search for hyperparameter tuning
        XGBmodel =  XGBClassifier(
            subsample=0.6,
            reg_lambda=1,
            reg_alpha=0.1,
            n_estimators=200,
            min_child_weight=3,
            max_depth=8,
            learning_rate=0.2,
            gamma=1,
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

        object_columns = stockData.select_dtypes(include="object").columns.tolist()
        features = X_train.columns.tolist()  
        encoder = OHEncoder  

        latest_data = stockData.iloc[-1]

        tomorrow_features = encode_new_row(latest_data, encoder, object_columns, features)
        tomorrow_features = tomorrow_features.astype(float) 


        tomorrow_pred = XGBmodel.predict(tomorrow_features)
        if tomorrow_pred[0] == 1:
            print("Predicted stock price will increase tomorrow.")
        else:
            print("Predicted stock price will decrease tomorrow.")
        findFeatureImportance(XGBmodel, features)
        df_for_corr = X_train.copy()
        df_for_corr["PriceIncrease"] = y_train
        findCorrelation(df_for_corr, features, "PriceIncrease")

def encode_new_row(new_row: pd.Series, encoder, object_columns: list, features: list) -> pd.DataFrame:
    numeric_part = new_row.drop(labels=object_columns)
    if len(object_columns) == 0 or encoder is None:
        # No categorical columns to encode, just return numeric part as dataframe with features order
        df = pd.DataFrame([numeric_part.values], columns=numeric_part.index)
        df = df.reindex(columns=features, fill_value=0)
        return df

    object_part = new_row[object_columns].values.reshape(1, -1)
    encoded_part = encoder.transform(object_part)
    encoded_df = pd.DataFrame(encoded_part, columns=encoder.get_feature_names_out(object_columns))
    final_df = pd.concat([pd.DataFrame([numeric_part.values], columns=numeric_part.index), encoded_df], axis=1)
    final_df = final_df.reindex(columns=features, fill_value=0)
    return final_df



def prepareRegressionData():
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

    OImputer = SimpleImputer(strategy="most_frequent")
    NImputer = SimpleImputer(strategy="mean")
    object_columns = stockData.select_dtypes(include="object").columns
    numeric_columns = stockData.select_dtypes(exclude="object").columns

    NImputed_X_Train = pd.DataFrame(NImputer.fit_transform(X_train[numeric_columns]))
    OImputed_X_Train = pd.DataFrame(OImputer.fit_transform(X_train[object_columns].astype(str)))
    NImputed_X_val = pd.DataFrame(NImputer.transform(X_val[numeric_columns]))
    OImputed_X_val = pd.DataFrame(OImputer.transform(X_val[object_columns].astype(str)))

    Imputed_X_Train = pd.concat([OImputed_X_Train, NImputed_X_Train], axis=1)
    Imputed_X_val = pd.concat([OImputed_X_val, NImputed_X_val], axis=1)

    #Re-add Columns

    Imputed_X_Train.columns = X_train.columns
    Imputed_X_val.columns = X_val.columns

     # Add One-Hot Encoding for Categorical Features
    
    print(f"Object columns to be one-hot encoded: {object_columns.tolist()}")

    OHEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    OHtrain = pd.DataFrame(OHEncoder.fit_transform(Imputed_X_Train[object_columns]),columns=OHEncoder.get_feature_names_out(object_columns))
    OHval = pd.DataFrame(OHEncoder.transform(Imputed_X_val[object_columns]), columns=OHEncoder.get_feature_names_out(object_columns))

    OHtrain.index = X_train.index
    OHval.index = X_val.index

    num_X_train = X_train.drop(object_columns, axis=1)
    num_X_val = X_val.drop(object_columns, axis=1)

    X_train = pd.concat([num_X_train, OHtrain], axis=1)
    X_val = pd.concat([num_X_val, OHval], axis=1)  
    features = X_train.columns.tolist()

    scoreDataset(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, task="classification", stockData=stockData, features=features, OHEncoder=OHEncoder)


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
    #prepareClassificationData()
    prepareRegressionData()

if __name__ == "__main__":
    main()

