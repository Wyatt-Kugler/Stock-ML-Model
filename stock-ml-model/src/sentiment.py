from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


def analyze_sentiment(combined_posts: pd.DataFrame) -> pd.DataFrame:
    # Analyze sentiment

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

    return dailySentiment
