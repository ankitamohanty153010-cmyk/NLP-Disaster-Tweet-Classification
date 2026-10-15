# NLP Disaster Tweet Classification

## Project Overview
This project uses Natural Language Processing (NLP) and Machine Learning to classify tweets as disaster-related or non-disaster-related.

The application allows users to enter a tweet and receive a predicted category along with model-estimated scores.

## Objectives
- Clean and prepare tweet text for analysis.
- Convert text into numerical features using TF-IDF.
- Train and evaluate machine-learning classification models.
- Deploy an interactive web application for tweet classification.

## Dataset
The project uses a labelled disaster-tweet dataset containing tweet text and target labels.

- `0` — Non-disaster tweet
- `1` — Disaster-related tweet

The dataset used contains 7,613 records.

## Methodology
1. Exploratory Data Analysis (EDA)
2. Text cleaning and preprocessing
3. TF-IDF feature extraction
4. Model training and comparison
5. Model evaluation using classification metrics
6. Saving the trained model and vectorizer
7. Deploying the application using Streamlit

## Model Evaluation
The balanced Logistic Regression model was selected for the application because it achieved a higher recall for disaster-related tweets than the original Logistic Regression model.

- Accuracy: 80.37%
- Disaster-class precision: 76.93%
- Disaster-class recall: 77.52%
- Disaster-class F1-score: 77.23%

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF
- Streamlit
- Joblib

## Live Application
[Open the Disaster Tweet Classification App](https://ankitamohanty153010-cmyk-nlp-disaster-tweet-classifi-app-g6mijx.streamlit.app)

## Repository Contents
- `NLP-Disaster-Tweet-Classification.ipynb` — Project notebook
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `models/` — Saved model and TF-IDF vectorizer

## Disclaimer
This model provides predictions based on patterns learned from labelled tweets. Its predictions and scores are not verified emergency reports and should not be used as the sole basis for emergency decisions.
