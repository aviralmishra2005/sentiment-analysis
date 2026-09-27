import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================
# PATHS
# ============================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Multilingual Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)
# ============================================
# MODEL LOADING
# ============================================

@st.cache_resource
def load_distilbert():
    model_path = MODEL_DIR / "distilbert_sentiment"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.eval()

    return tokenizer, model


@st.cache_resource
def load_lstm():
    model = load_model(
        MODEL_DIR / "lstm_sentiment.keras"
    )

    with open(MODEL_DIR / "lstm_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


@st.cache_resource
def load_classical_models():

    vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )

    logistic_model = joblib.load(
        MODEL_DIR / "logistic_regression.joblib"
    )

    svm_model = joblib.load(
        MODEL_DIR / "linear_svm.joblib"
    )

    nb_model = joblib.load(
        MODEL_DIR / "naive_bayes.joblib"
    )

    return vectorizer, logistic_model, svm_model, nb_model
# ============================================
# DISTILBERT SENTIMENT PREDICTION
# ============================================

def predict_distilbert(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0, predicted_class
    ].item()

    if predicted_class == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, confidence
# ============================================
# SIDEBAR NAVIGATION
# ============================================

st.sidebar.title("💬 Sentiment AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Sentiment Analyzer",
        "Model Comparison"
    ]
)
# ============================================
# HOME PAGE
# ============================================

if page == "Home":

    st.title("💬 Multilingual Sentiment Analysis System")

    st.markdown(
        """
        ### AI-powered text sentiment analysis

        This project analyzes text and predicts whether the expressed
        sentiment is **Positive** or **Negative**.

        The system was developed by comparing multiple machine-learning
        and deep-learning approaches, including:

        - Naive Bayes
        - Linear SVM
        - Logistic Regression
        - LSTM
        - DistilBERT

        ### 🔬 Project Workflow

        ```text
        Input Text
             ↓
        Text Tokenization
             ↓
        DistilBERT
             ↓
        Sentiment Classification
             ↓
        Positive / Negative
             ↓
        Confidence Score
        ```

        ### 📊 Dataset

        The models were trained using the **Sentiment140** dataset.

        ### 🎯 Main Objective

        To compare traditional machine-learning techniques with
        deep-learning and Transformer-based NLP models for sentiment
        classification.
        """
    )

    st.info(
        "Use the sidebar to open the Sentiment Analyzer "
        "or view the Model Comparison."
    )
# ============================================
# SENTIMENT ANALYZER
# ============================================

elif page == "Sentiment Analyzer":

    st.title("🔍 Sentiment Analyzer")

    st.write(
        "Enter a sentence or short piece of text and the AI model "
        "will analyze its sentiment."
    )

    text_input = st.text_area(
        "Enter your text:",
        placeholder="Example: I really enjoyed this movie!",
        height=150
    )

    analyze_button = st.button(
        "Analyze Sentiment",
        type="primary"
    )

    if analyze_button:

        if not text_input.strip():

            st.warning("Please enter some text first.")

        else:

            with st.spinner("Analyzing sentiment..."):

                tokenizer, model = load_distilbert()

                sentiment, confidence = predict_distilbert(
                    text_input,
                    tokenizer,
                    model
                )

            st.subheader("Analysis Result")

            if sentiment == "Positive":
                st.success(f"😊 {sentiment}")
            else:
                st.error(f"😞 {sentiment}")

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(confidence)

            st.caption(
                "Prediction generated using the fine-tuned DistilBERT model."
            )
# ============================================
# MODEL COMPARISON
# ============================================

elif page == "Model Comparison":

    st.title("📊 Model Comparison")

    st.write(
        "Performance comparison of the machine-learning, "
        "deep-learning, and Transformer-based models."
    )

    # Load saved results
    validation_results = pd.read_csv(
        MODEL_DIR / "validation_model_comparison.csv"
    )

    test_results = pd.read_csv(
        MODEL_DIR / "test_model_comparison.csv"
    )

    # ----------------------------------------
    # Validation Results
    # ----------------------------------------

    st.subheader("Validation Set Performance")

    st.dataframe(
        validation_results,
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------------------
    # Test Results
    # ----------------------------------------

    st.subheader("Final Test Set Performance")

    st.dataframe(
        test_results,
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------------------
    # Accuracy Comparison
    # ----------------------------------------

    st.subheader("Accuracy Comparison")

    chart_data = validation_results.set_index("Model")[
        "Accuracy"
    ]

    st.bar_chart(chart_data)

    st.caption(
        "The validation results are based on the held-out validation "
        "set used during model development. The test results are from "
        "the untouched final test set."
    )
# ============================================
# FOOTER
# ============================================

st.sidebar.markdown("---")
st.sidebar.caption(
    "Multilingual Sentiment Analysis System | B.Tech Major Project"
)