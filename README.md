# 🎬 IMDB Movie Review Sentiment Analysis

This project is a **Machine Learning application** that predicts the **sentiment** (Positive / Negative) of IMDB movie reviews using **Logistic Regression with TF-IDF vectorization**.

The app is built with **Streamlit** and can be deployed on **Streamlit Cloud**.

---

## 📌 Features

- ✅ Enter a **movie review** and get instant sentiment prediction  
- ✅ Uses **TF-IDF Vectorizer + Logistic Regression**  
- ✅ Lightweight and easy to deploy  
- ✅ Can be extended to batch CSV review prediction  

---

## 🗂️ Project Structure

imdb-sentiment-app/
│
├── app.py # Streamlit app
├── imdb_sentiment_model.pkl # Saved ML model
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── requirements.txt # Python dependencies
└── README.md # Project documentation

📊 Model Details
Algorithm: Logistic Regression

Text Representation: TF-IDF (max_features=5000, stop_words='english')

Training Data: IMDB 50K Movie Reviews Dataset

Accuracy: ~88-90% on test set
