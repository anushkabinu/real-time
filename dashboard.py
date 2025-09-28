import streamlit as st
import pandas as pd
import requests
import time
import altair as alt
import nltk

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ----------------------------
# Setup
# ----------------------------
nltk.download("vader_lexicon", quiet=True)

st.set_page_config(page_title="AI News Sentiment", page_icon="📰", layout="wide")
st.title("📰 News Sentiment Dashboard (PySpark + VADER)")

# ----------------------------
# Spark Session (cached)
# ----------------------------
@st.cache_resource
def init_spark():
    return (
        SparkSession.builder
        .appName("NewsSentimentApp")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

spark = init_spark()

# ----------------------------
# API Key (placeholder)
# ----------------------------
API_KEY = "c4a6a67abbd943b7b63aa7f833cef51b"  # <-- Replace with your NewsAPI key


# ----------------------------
# Data Fetching
# ----------------------------
def get_news(query: str, limit: int = 5) -> pd.DataFrame:
    """Fetch latest news headlines using NewsAPI."""
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={query}&pageSize={limit}&sortBy=publishedAt&apiKey={API_KEY}"
    )
    resp = requests.get(url).json()
    articles = resp.get("articles", [])
    return pd.DataFrame([{"headline": a["title"]} for a in articles if a.get("title")])


# ----------------------------
# Sentiment Labeling (VADER)
# ----------------------------
def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment labels using VADER sentiment analysis."""
    sia = SentimentIntensityAnalyzer()

    def classify(text: str) -> str:
        score = sia.polarity_scores(text)["compound"]
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        return "Neutral"

    df["sentiment"] = df["headline"].apply(classify)
    df["label"] = df["sentiment"].map({"Positive": 2.0, "Neutral": 1.0, "Negative": 0.0})
    return df


# ----------------------------
# UI Inputs
# ----------------------------
query = st.text_input("Enter Topic", "Artificial Intelligence")
mode = st.radio("Choose Mode", ["Train Model", "Predict Sentiment"], horizontal=True)
limit = st.slider("Number of Articles", 3, 20, 8)

# ----------------------------
# Main Logic
# ----------------------------
if st.button("Run Analysis"):
    news_df = get_news(query, limit)

    if news_df.empty:
        st.warning("No news articles found. Try another topic.")
    else:
        # ---------- Training ----------
        if mode == "Train Model":
            news_df = add_sentiment(news_df)
            spark_df = spark.createDataFrame(news_df)

            # Text Processing
            tokenizer = Tokenizer(inputCol="headline", outputCol="tokens")
            tokenized = tokenizer.transform(spark_df)

            hashing = HashingTF(inputCol="tokens", outputCol="tf")
            tf_data = hashing.transform(tokenized)

            idf = IDF(inputCol="tf", outputCol="features").fit(tf_data)
            final_data = idf.transform(tf_data)

            # Model Training
            lr = LogisticRegression(maxIter=10, regParam=0.001)
            model = lr.fit(final_data)

            # Evaluation
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="accuracy"
            )
            accuracy = evaluator.evaluate(model.transform(final_data))

            # Save Model + Pipeline
            st.session_state.update(
                {"model": model, "tokenizer": tokenizer, "hashing": hashing, "idf": idf}
            )

            # Display Results
            st.success(f"Model trained successfully ✅ | Accuracy: {accuracy:.2%}")
            st.dataframe(news_df[["headline", "sentiment"]])

        # ---------- Prediction ----------
        elif mode == "Predict Sentiment":
            if "model" not in st.session_state:
                st.error("⚠️ Please train the model first!")
            else:
                # Retrieve pipeline + model
                tok = st.session_state["tokenizer"]
                hash_tf = st.session_state["hashing"]
                idf_model = st.session_state["idf"]
                model = st.session_state["model"]

                # Transform new data
                spark_df = spark.createDataFrame(news_df)
                tokenized = tok.transform(spark_df)
                tf_data = hash_tf.transform(tokenized)
                final_data = idf_model.transform(tf_data)
                preds = (
                    model.transform(final_data)
                    .select("headline", "prediction")
                    .toPandas()
                )

                # Map back labels
                reverse_map = {2.0: "Positive", 1.0: "Neutral", 0.0: "Negative"}
                preds["sentiment"] = preds["prediction"].map(reverse_map)

                # Show Results
                st.subheader("Predicted Sentiments")
                for _, row in preds.iterrows():
                    st.markdown(f"✅ *{row['headline']}* → {row['sentiment']}")
                    time.sleep(0.5)

                # Visualization
                st.subheader("Sentiment Distribution")
                chart_data = preds.groupby("sentiment").size().reset_index(name="count")
                chart = (
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("sentiment", sort=["Positive", "Neutral", "Negative"]),
                        y="count",
                        color="sentiment",
                    )
                )
                st.altair_chart(chart, use_container_width=True)
