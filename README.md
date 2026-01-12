# Explainability_Fake-News-Detection-Project
Explainable Fake News Detection using Classical NLP &amp; Machine Learning

# 📰 Explainable Fake News Detection using NLP

An end-to-end **Explainable NLP system** that predicts whether a news article is **Fake or Real** using classical machine learning techniques, with a strong focus on **interpretability, responsible AI, and clean deployment**.

---

## 🔍 Project Overview

Fake news poses a serious threat to public trust and information integrity.  
This project builds a **transparent and explainable fake news detection system** that not only predicts the credibility of news articles but also explains **why** a particular prediction was made.

The system is trained using **TF-IDF features** and a **Linear Support Vector Machine (SVM)** and is deployed via a **Streamlit web application** with a clean and user-friendly interface.

---

## 🚀 Key Features

- 🧠 **Machine Learning-based Fake News Classification**
- 🔎 **Explainability using SHAP (offline analysis)**
- 📊 **Lightweight interpretability in deployment**
- 🌐 **Interactive Streamlit Web App**
- ⚖️ **Responsible AI considerations**
- 📦 **Clean separation of training and inference**

---

## 🧪 Dataset

The model is trained on a publicly available fake news dataset consisting of:
- Fake news articles
- Real news articles (primarily Reuters-style reporting)

During deployment, **no dataset is loaded** — only the trained model and vectorizer are used.

---

## 🧠 Methodology

### 1️⃣ Text Preprocessing
- Lowercasing
- URL removal
- HTML tag removal
- Removal of non-alphabetic characters

> ⚠️ Explicit lemmatization and stopword removal were intentionally avoided to preserve stylistic and surface-level linguistic cues, which are important for fake news detection.

---

### 2️⃣ Feature Engineering
- **TF-IDF Vectorization**
- Unigrams and Bigrams
- Controlled vocabulary size for better generalization

Tokenization and n-gram generation are handled internally by the TF-IDF vectorizer.

---

### 3️⃣ Model Training
- **Linear Support Vector Machine (SVM)**
- Evaluated using **F1-score**, prioritizing balanced precision and recall
- Achieved strong generalization with minimal train–test performance gap

---

### 4️⃣ Explainability
- **SHAP (SHapley Additive exPlanations)** used during analysis
- Word-level feature attribution to understand model decisions
- Confirmed presence of dataset-specific linguistic patterns and biases

> SHAP is used during the research phase and not computed live in the deployed app to ensure fast inference and better UX.

---

## 🌐 Streamlit Web Application

The deployed application allows users to:
1. Paste a news article
2. Get a **Fake / Real** prediction
3. View **key influential words** contributing to the decision
4. Explore model details via an expandable section

The app focuses on **clarity, speed, and interpretability**.

---

## 🖥️ Project Structure

         Explainable_Fake_News/
        │
        ├── app.py # Streamlit application
        ├── models/
        │ ├── model.pkl # Trained SVM model
        │ ├── vectorizer.pkl # TF-IDF vectorizer
        ├── requirements.txt
        └── README.md

### Install dependencies
        pip install -r requirements.txt

### Run the Streamlit app
       streamlit run app.py

## Check the Project at -
    Link: https://explainabilityfake-news-detection-project-7pjpsyhkbm6xcfmsmnv2.streamlit.app/

### Responsible AI Considerations

-  Predictions are based on learned linguistic patterns

-  Model may reflect dataset-specific biases

-  Outputs should not be treated as definitive judgments

-  Intended strictly for educational and research purposes

## Future Improvements

- Cross-dataset evaluation for robustness

- Comparison with lemmatized pipelines

- Transformer-based models with explainability

- Confidence calibration for predictions

### 📜 License

This project is released for educational and research purposes only.

👤 Author

Pranav Mishra
NLP Research-oriented Project | IIT Patna
