# 📰 Fake News Detector – WELFake Dataset

Hi! 👋 I'm a passionate data science student who loves turning messy real-world data into models that actually make a difference.

This project is my attempt at building a **reliable, fast, and interpretable fake news classifier** using classic NLP techniques — because fighting misinformation is one of the most meaningful things we can do with data science today.

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-96.4%25-brightgreen?style=for-the-badge" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.9932-blueviolet?style=for-the-badge" alt="ROC-AUC"/>
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-orange?style=for-the-badge" alt="Model"/>
</p>

## 🙌 Why I built this

Fake news spreads fast and harms trust in information.  
I wanted to prove that **even with classic ML (no transformers, no GPU)**, we can build something really effective — and make it accessible to everyone.

This project taught me a ton about:
- End-to-end ML workflows
- Importance of proper text cleaning
- Value of hyperparameter tuning
- How to productionize models (save vectorizer + model!)
- Building user-friendly interfaces with Streamlit

## 🎯 What this project does

Given a news title and/or article text, the model predicts whether it's **likely real** or **likely fake**.

- Training accuracy (cross-validated): ~96.5%  
- Test set accuracy: **96.39%**  
- ROC-AUC: **0.9932** → almost perfect discrimination  
- Very balanced performance on both classes (F1 ~0.96 for real & fake)

## ✨ Features

- Clean, reproducible pipeline from raw CSV → production-ready model
- Proper text preprocessing (stopwords, lemmatization, URL/mention removal)
- TF-IDF with bigrams (`ngram_range=(1,2)`) – captures phrases very well
- Hyperparameter tuning with **GridSearchCV**
- Model & vectorizer saved with `joblib` → ready for deployment
- Simple but powerful Streamlit web app demo
- Visual EDA + error analysis + ROC curve interpretation

## 🛠️ Tech Stack

- **Language**: Python  
- **Data Handling**: pandas, numpy  
- **NLP**: nltk (lemmatization, stopwords)  
- **Vectorization**: scikit-learn TfidfVectorizer  
- **Modeling**: Logistic Regression + GridSearchCV  
- **Evaluation**: classification_report, ROC-AUC, confusion matrix  
- **Visualization**: matplotlib, seaborn  
- **Model Persistence**: joblib  
- **Web App**: Streamlit (local + ready for cloud)

## 📊 Results at a Glance

| Metric              | Value     | Comment                             |
|---------------------|-----------|-------------------------------------|
| Test Accuracy       | 96.39%    | Very strong for TF-IDF + LR         |
| Fake News Recall    | 97.16%    | Excellent at catching misinformation|
| ROC-AUC             | 0.9932    | Near-perfect separability           |
| 5-Fold CV Accuracy  | 96.53% ± 0.20% | Stable across folds             |

## 🚀 Try it yourself!

### 1. Local demo

```bash
# Clone the repo
git clone https://github.com/YOUR-USERNAME/fake-news-detector.git
cd fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### 2. Deployment on Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and select this repository.
4. Select `app.py` as the main file.
5. Click **Deploy**.

## 🔮 What's Next? (Future Improvements)

I'm always learning! Here is what I plan to add next:
- [ ] Experiment with **LSTM/GRU** (Deep Learning) for better context understanding.
- [ ] Add a feature to **scrape URLs** directly instead of pasting text.
- [ ] Improve the dataset with more recent news articles.

---

### 🤝 Let's Connect!

I love talking about Data & AI. If you have any feedback or just want to chat, feel free to reach out!

Happy Coding! 💻✨
