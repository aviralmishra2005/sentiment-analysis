# 💬 Sentiment Analysis System

A machine-learning and deep-learning based sentiment analysis system developed as a B.Tech major project.

The project compares traditional machine-learning algorithms with an LSTM neural network and a Transformer-based DistilBERT model for binary sentiment classification using the Sentiment140 dataset.

The trained models are integrated into an interactive Streamlit web application for real-time sentiment prediction.

---

## 📌 Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task that determines the emotional polarity expressed in a piece of text.

This project classifies text into:

- Positive
- Negative

Multiple approaches were implemented and compared to study the progression from traditional machine learning to deep learning and Transformer-based NLP.

### Models Used

1. Naive Bayes
2. Linear SVM
3. Logistic Regression
4. LSTM
5. DistilBERT

---

## 🎯 Objectives

The main objectives of the project are:

- Perform text preprocessing and exploratory data analysis.
- Convert text into numerical representations using TF-IDF.
- Implement traditional machine-learning sentiment classifiers.
- Develop an LSTM-based sentiment classifier.
- Fine-tune a Transformer-based DistilBERT model.
- Compare different approaches using standard classification metrics.
- Evaluate the models on an untouched test dataset.
- Deploy the trained models through a Streamlit web application.

---

## 📊 Dataset

The project uses the **Sentiment140** dataset.

The dataset contains text sentences and their corresponding binary sentiment labels.

### Dataset Structure

```text
data/
└── sentiment140/
    ├── train_data.csv
    ├── test_data.csv
    ├── vocab.json
    └── vocab.py
