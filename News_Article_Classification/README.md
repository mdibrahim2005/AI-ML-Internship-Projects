# 📰 AI News Article Classification

A machine learning web application that classifies news articles as **Fake News** or **Real News** using **TF-IDF Vectorization** and **Logistic Regression**.

The project provides a simple Flask-based web interface where users can enter a news headline or article and receive a classification along with the model's probability for each class.

---

## 🚀 Project Overview

Fake news can spread quickly through online platforms and social media.

This project demonstrates how Natural Language Processing (NLP) and Machine Learning can be used to analyze the textual patterns of news articles and classify them into two categories:

- 🚨 Fake News
- ✅ Real News

The application uses a trained Logistic Regression model together with TF-IDF text features.

---

## 🎯 Objective

The main objectives of this project are:

- Process and analyze news article text.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classification model.
- Evaluate the model using a separate test dataset.
- Build a Flask web application for real-time predictions.
- Display the predicted class and model probabilities through a simple web interface.

---

## 🧠 Machine Learning Workflow

The project follows this workflow:

```text
News Dataset
     ↓
Data Preparation
     ↓
Combine Title + Article Text
     ↓
Train/Test Split
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
Save Model
     ↓
Flask Web Application
     ↓
User Enters News
     ↓
Fake / Real Prediction