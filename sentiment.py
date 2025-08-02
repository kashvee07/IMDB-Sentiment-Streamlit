import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("🎬Movie Review Sentiment Analysis")
st.write("Enter a movie review below and check if it's positive or negative!")

# Text input
user_review = st.text_area("Write your review here:")

if st.button("Predict Sentiment"):
    if user_review.strip() != "":
        # Transform review to TF-IDF
        review_tfidf = vectorizer.transform([user_review])
        
        # Predict sentiment
        prediction = model.predict(review_tfidf)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😡"
        
        st.subheader(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review first.")
