from config import CACHE_FILE
from reddit_utils import fetch_reddit_posts
from sentiment import analyze_sentiment
from data_preparation import prepareRegressionData, prepareClassificationData
import os
import pandas as pd
import time

def main():
    if os.path.exists(CACHE_FILE):
        print("Loading cached Reddit posts...")
        combined_posts = pd.read_csv(CACHE_FILE, parse_dates=["Date"])
    else:
        subreddits = ["investing", "stocks", "wallstreetbets", "StockMarket", "financialindependence", "dividends", "pennystocks", "options", "algotrading"]
        timeframes = ["year", "all"]
        sortvalues = ["top", "hot"]
        all_posts = []
        for subreddit in subreddits:
            for timeframe in timeframes:
                for sortval in sortvalues:
                    df = fetch_reddit_posts(subreddit, "microsoft", timeframe, sortval)
                    all_posts.append(df)
                    time.sleep(1)
        combined_posts = pd.concat(all_posts).drop_duplicates().reset_index(drop=True)
        combined_posts.to_csv(CACHE_FILE, index=False)

    sentiment_df = analyze_sentiment(combined_posts)
    #prepareRegressionData(sentiment_df)
    prepareClassificationData(sentiment_df)

if __name__ == "__main__":
    main()
# This script is the main entry point for the stock ML model, which fetches Reddit posts,
# analyzes their sentiment, and prepares the data for regression analysis.
# It uses caching to avoid repeated API calls and speeds up the process.
# The sentiment analysis is done using the VADER sentiment analysis tool.
# The final data preparation step is handled by the prepareRegressionData function.