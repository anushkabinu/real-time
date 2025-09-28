import streamlit as st
import pandas as pd
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import altair as alt
import time

# Download VADER lexicon
nltk.download("vader_lexicon", quiet=True)

# API Key for NewsAPI
API_KEY = "c4a6a67abbd943b7b63aa7f833cef51b"  # Add your NewsAPI key here

# Function to fetch news articles
def fetch_news(topic: str, limit: int = 5) -> pd.DataFrame:
    url = f"https://newsapi.org/v2/everything?q={topic}&pageSize={limit}&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url).json()
    articles = response.get("articles", [])
    data = [{"headline": a["title"]} for a in articles if a.get("title")]
    return pd.DataFrame(data)

# Sentiment analysis
def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    
    def sentiment_label(text):
        score = sia.polarity_scores(text)["compound"]
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    df["sentiment"] = df["headline"].apply(sentiment_label)
    return df

# Streamlit layout
st.set_page_config(page_title="AI News Sentiment", page_icon="📰", layout="wide")
st.title("📰 AI News Sentiment Explorer")

with st.sidebar:
    st.header("Settings")
    topic = st.text_input("Enter Topic", "Artificial Intelligence")
    num_articles = st.slider("Number of Articles", min_value=3, max_value=20, value=8)
    mode = st.radio("Mode", ["View Sentiment", "Train Dummy Model"], horizontal=True)

# Fetch news button
if st.button("Analyze News"):
    news_df = fetch_news(topic, num_articles)
    
    if news_df.empty:
        st.warning("No articles found for this topic.")
    else:
        news_df = analyze_sentiment(news_df)
        
        # Display results
        st.subheader(f"News Headlines on '{topic}'")
        for idx, row in news_df.iterrows():
            icon = "✅" if row["sentiment"] == "Positive" else "⚠️" if row["sentiment"] == "Neutral" else "❌"
            st.markdown(f"{icon} **{row['headline']}** → *{row['sentiment']}*")
            time.sleep(0.3)

        # Chart sentiment distribution
        st.subheader("Sentiment Overview")
        chart_data = news_df["sentiment"].value_counts().reset_index()
        chart_data.columns = ["sentiment", "count"]
        
        chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("sentiment", sort=["Positive", "Neutral", "Negative"]),
            y="count",
            color="sentiment",
            tooltip=["sentiment", "count"]
        ).properties(height=400)
        
        st.altair_chart(chart, use_container_width=True)

        if mode == "Train Dummy Model":
            st.info("Training is skipped in this simplified version. VADER scores are used directly for sentiment.")
