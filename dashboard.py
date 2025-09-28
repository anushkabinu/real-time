import streamlit as st
import time
import matplotlib.pyplot as plt

st.title("📰 Real-Time News Sentiment Dashboard")

while True:
    # Dummy data for demo
    predictions = [{"text": "Market is rising!", "prediction": 1.0},
                   {"text": "Stocks crash badly", "prediction": 0.0}]
    
    import pandas as pd
    df = pd.DataFrame(predictions)
    df["sentiment"] = df["prediction"].apply(lambda x: "Positive" if x == 1.0 else "Negative")
    
    st.write(df[["text", "sentiment"]])
    
    sentiment_counts = df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)
    
    time.sleep(30)
