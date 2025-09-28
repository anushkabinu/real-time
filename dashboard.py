# dashboard.py
import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from newsapi import NewsApiClient
import os
import time

# -------------------------------
# 1️⃣ Initialize Spark Session
# -------------------------------
spark = SparkSession.builder.appName("NewsSentimentDashboard").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# -------------------------------
# 2️⃣ News API setup
# -------------------------------
# You can also set your key as an environment variable for security
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "c4a6a67abbd943b7b63aa7f833cef51b")
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def fetch_news():
    """Fetch top headlines"""
    try:
        articles = newsapi.get_top_headlines(language='en', page_size=20)['articles']
        return [a['title'] for a in articles if a['title']]
    except Exception as e:
        print("Error fetching news:", e)
        return []

# -------------------------------
# 3️⃣ Train or load a PySpark ML sentiment model
# -------------------------------
# For simplicity, we train a small example model on dummy data
training = spark.createDataFrame([
    (0, "Markets crash due to recession fears", "negative"),
    (1, "Company profits reach record high", "positive"),
    (2, "New breakthrough in cancer research", "positive"),
    (3, "War escalates in conflict zone", "negative"),
], ["id", "text", "label"])

indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
tf = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="labelIndex")

pipeline = Pipeline(stages=[indexer, tokenizer, remover, tf, idf, lr])
model = pipeline.fit(training)

# -------------------------------
# 4️⃣ Streamlit Dashboard
# -------------------------------
st.title("📰 Real-Time News Sentiment Dashboard")

st.write("Fetching latest news and classifying sentiment...")

# Auto-refresh every 30 seconds
refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 30)

while True:
    news_batch = fetch_news()
    if news_batch:
        # Convert to Spark DataFrame
        df_spark = spark.createDataFrame([(i, text) for i, text in enumerate(news_batch)], ["id", "text"])
        preds = model.transform(df_spark).select("text", "prediction").toPandas()
        preds['sentiment'] = preds['prediction'].apply(lambda x: "Positive" if x == 1.0 else "Negative")

        # Display table
        st.write(preds[['text', 'sentiment']])

        # Plot sentiment counts
        sentiment_counts = preds['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

    else:
        st.write("No news fetched. Check your API key or internet connection.")

    time.sleep(refresh_interval)
