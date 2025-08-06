import os
from dotenv import load_dotenv

# Update this path to where your Api.env file is
load_dotenv(dotenv_path="c:/Users/wyatt/Downloads/Progamming projets/stock-ml-model/src/Api.env")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

CACHE_FILE = "reddit_posts_cache.csv"
TICKER = "MSFT"
