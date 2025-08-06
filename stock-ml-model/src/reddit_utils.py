import praw
from datetime import datetime, timedelta
import pandas as pd
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

def fetch_reddit_posts(subreddit, keyword, timeframe, sortvalue):
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
