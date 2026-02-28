import streamlit as st
import joblib
import re
import math

# Load model
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# UI
st.title("📰 Fake News Detector for Students")

news = st.text_area("Enter News Article")

if st.button("Check News"):

    if news.strip() == "":
        st.warning("Please enter news text!")
    else:
        news_clean = clean_text(news)
        news_vector = vectorizer.transform([news_clean])

        prediction = model.predict(news_vector)[0]

        score = model.decision_function(news_vector)[0]
        confidence = 1/(1+math.exp(-score))
        confidence = round(confidence*100,2)

        if prediction == "REAL":
            st.success(f"✅ REAL NEWS ({confidence}%)")
        else:
            st.error(f"⚠️ FAKE NEWS ({confidence}%)")
