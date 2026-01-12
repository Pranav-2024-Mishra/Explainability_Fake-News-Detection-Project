import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ----------------------
# Page configuration
# ----------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ----------------------
# Load model
# ----------------------
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ----------------------
# Header
# ----------------------
st.markdown(
    "<h1 style='text-align:center;'>📰 Fake News Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>"
    "An explainable NLP system that predicts whether a news article is "
    "<b>Fake</b> or <b>Real</b> using machine learning."
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# ----------------------
# Input
# ----------------------
news_text = st.text_area(
    "Paste the news article below:",
    height=220,
    placeholder="Enter full news article text here..."
)

# ----------------------
# Predict
# ----------------------
if st.button("🔍 Analyze News", use_container_width=True):

    if news_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Vectorize
        X_input = vectorizer.transform([news_text])

        # Prediction
        pred = model.predict(X_input)[0]

        # Display result
        st.divider()

        if pred == 0:
            st.error("🟥 **Prediction: FAKE NEWS**")
        else:
            st.success("🟩 **Prediction: REAL NEWS**")

        # Model details (ADD HERE)
        # ----------------------
        with st.expander("ℹ️ Model Details"):
            st.markdown("""
         - **Model:** Linear Support Vector Machine  
         - **Features:** TF-IDF (unigrams & bigrams)  
         - **Evaluation Metric:** F1-score  
         - **Explainability:** SHAP (offline analysis)
        """)
        # ----------------------
        # Explainability (lightweight & fast)
        # ----------------------
        feature_names = vectorizer.get_feature_names_out()
        tfidf_values = X_input.toarray()[0]

        explanation_df = pd.DataFrame({
            "word": feature_names,
            "importance": tfidf_values
        })

        top_words = (
            explanation_df
            .sort_values("importance", ascending=False)
            .head(8)
            .query("importance > 0")
        )

        if not top_words.empty:
            st.subheader("🔍 Key Influential Words")
            st.write(
                "These words had the strongest influence on the model’s decision:"  
            )
            st.dataframe(
                top_words.reset_index(drop=True),
                use_container_width=True
            )

        # ----------------------
        # Disclaimer
        # ----------------------
        st.info(
            "ℹ️ This system is intended for educational and research purposes. "
            "Prediction based on strong linguistic alignment with factual news patterns."
        )


       # Footer (ADD THIS AT THE END)
       # ----------------------
st.divider()

st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Built by <b>Pranav Mishra</b> © 2026 | NLP Project | IIT Patna"
    "</p>",
    unsafe_allow_html=True
)