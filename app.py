# ============================================
# NLP PROJECT: DISASTER TWEET CLASSIFICATION
# File: app.py
# Purpose: Create a web interface for tweet prediction
# ============================================

import streamlit as st
import joblib
import re
import warnings
from pathlib import Path

# Suppress InconsistentVersionWarning from scikit-learn model loading
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# --------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------

st.set_page_config(
    page_title="Disaster Tweet Classifier",
    page_icon="🚨",
    layout="centered"
)

# --------------------------------------------
# 2. LOAD THE SAVED MODEL AND TF-IDF VECTORIZER
# --------------------------------------------

MODEL_FOLDER = Path(__file__).parent / "models"

model_path = MODEL_FOLDER / "disaster_tweet_model.pkl"
vectorizer_path = MODEL_FOLDER / "tfidf_vectorizer.pkl"

@st.cache_resource
def load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# --------------------------------------------
# 3. CLEAN THE INPUT TWEET
# --------------------------------------------

def clean_tweet(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# --------------------------------------------
# 4. CREATE THE WEB PAGE
# --------------------------------------------

st.title("🚨 Disaster Tweet Classification")

st.write(
    "Enter a tweet below to predict whether it is "
    "related to a disaster."
)

st.info(
    "This application uses a machine-learning model "
    "trained on labelled tweets."
)

# --------------------------------------------
# 5. ACCEPT TWEET INPUT AND PREDICT
# --------------------------------------------

tweet = st.text_area(
    "Enter your tweet:",
    placeholder="Example: A massive earthquake has hit the city...",
    height=120
)

if st.button("Classify Tweet"):
    if not tweet.strip():
        st.warning("Please enter a tweet first.")

    else:
        try:
            # Load the saved model and vectorizer
            model, vectorizer = load_model()

            # Clean the input tweet
            cleaned_tweet = clean_tweet(tweet)

            # Convert the tweet into TF-IDF features
            tweet_features = vectorizer.transform([cleaned_tweet])

            # Predict the class
            prediction = int(model.predict(tweet_features)[0])

            # Get model-estimated scores for both classes
            scores = model.predict_proba(tweet_features)[0]

            non_disaster_score = float(scores[0]) * 100
            disaster_score = float(scores[1]) * 100

            # Display the prediction
            if prediction == 1:
                st.error("🚨 Prediction: Disaster-related tweet")
            else:
                st.success("✅ Prediction: Non-disaster tweet")

            # Display model scores
            st.subheader("Model Scores")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Non-disaster score",
                    f"{non_disaster_score:.1f}%"
                )

            with col2:
                st.metric(
                    "Disaster-related score",
                    f"{disaster_score:.1f}%"
                )

            st.caption(
                "These are model-estimated scores, not verified "
                "emergency reports."
            )

        except FileNotFoundError:
            st.error(
                "The saved model files were not found. "
                "Please check that both model files are inside "
                "the 'models' folder."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
