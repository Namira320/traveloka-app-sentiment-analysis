# ✈️ Traveloka App Sentiment Analysis & Classification

This repository contains a complete pipeline for scraping user reviews of the Traveloka application from the Google Play Store, performing text preprocessing and normalization in Indonesian, extracting features using TF-IDF, and training/evaluating multiple machine learning and deep learning models for sentiment classification.

---

## 🚀 Project Overview

The project follows a standard Natural Language Processing (NLP) and Machine Learning workflow:

1. **Data Collection**: Scraping user reviews using `google-play-scraper`.
2. **Text Preprocessing & Cleaning**: 
   * Case folding (converting text to lowercase).
   * Removing URLs, usernames, numbers, and punctuation using regex (`re.sub`).
   * Tokenization.
   * Stopword removal (including custom Indonesian slang/colloquial stop words such as `'yg'`, `'nya'`, `'di'`, `'sih'`, `'dong'`, etc.).
   * Stemming using the Sastrawi library.
3. **Labeling**: Mapping rating scores to sentiment classes (Negative, Neutral, Positive).
4. **Feature Extraction**: Converting text data into numerical feature vectors using `TfidfVectorizer`.
5. **Model Training & Evaluation**: Comparing three distinct classification schemes:
   * **Schema 1**: Naive Bayes (`MultinomialNB`)
   * **Schema 2**: Random Forest (`RandomForestClassifier`)
   * **Schema 3**: Deep Learning / Neural Network (`MLPClassifier`)
6. **Inference**: Testing unseen sample reviews using the best-performing model.

---

## 📂 Project Structure

```text
├── deep_learning_analisis_sentimen.ipynb   # Main Jupyter Notebook containing the full pipeline
├── dataset_ulasan_hotel_traveloka.csv       # Raw dataset
├── dataset_traveloka_bersih.csv             # Cleaned preprocessed dataset
└── requirements.txt                         # Python dependencies

```

---

## 🛠️ Prerequisites & Installation

To run this project, make sure you have Python installed along with the required libraries. You can install the necessary packages using the following command:

```bash
pip install google-play-scraper pandas Sastrawi nltk scikit-learn tqdm

```

Or install directly from the requirements file if available:

```bash
pip install -r requirements.txt

```

---

## 💻 Pipeline Steps & Implementation

### 1. Text Preprocessing & Cleaning

Raw reviews are cleaned to remove noise (links, symbols, numbers), followed by stopword removal and stemming to standardize Indonesian text:

* Case folding (lowercase conversion)
* Regex-based cleaning (`re.sub`)
* Stemming via `Sastrawi.Stemmer.StemmerFactory`
* Custom stopword expansion for slang/frequent colloquial Indonesian terms (`'yg'`, `'nya'`, `'di'`, `'sih'`, `'dong'`, etc.)

### 2. Labeling Strategy

Ratings are mapped into sentiment categories:

* Rating $\le 2$ $\rightarrow$ **Negative**
* Rating $= 3$ $\rightarrow$ **Neutral**
* Rating $> 3$ $\rightarrow$ **Positive**

### 3. Model Schemes & Performance

Data is split into 80% training and 20% testing sets with TF-IDF features (max features = 5000):

* **Schema 1 (Naive Bayes)**: Training Accuracy: ~93.63% | Testing Accuracy: ~92.37%
* **Schema 2 (Random Forest)**: Training Accuracy: ~98.99% | Testing Accuracy: ~92.37%
* **Schema 3 (Deep Learning - MLP)**: Training Accuracy: ~98.71% | Testing Accuracy: ~91.74%

---

## 🔍 Running Inference

You can test new review sentences using the trained Multi-Layer Perceptron (MLP) model as implemented in the final section of the notebook:

```python
# Example inference snippet
ulasan_baru = [
    "aplikasi ini sangat membantu, pesan tiket hotel gampang banget",
    "kecewa parah, minta refund susahnya minta ampun tidak ada kejelasan"
]

```
