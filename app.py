
import streamlit as st
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Configure page
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .real-news {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .fake-news {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .confidence-score {
        font-size: 24px;
        font-weight: bold;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Application Title
st.title("📰 Fake News Detector")
st.markdown("Enter a news headline or article below to check its authenticity using machine learning.")

# Sidebar - About
with st.sidebar:
    st.header("About")
    st.info("""
    This app uses a **Logistic Regression** model trained on a dataset of real and fake news.
    
    **Features:**
    - TF-IDF Vectorization
    - Text Preprocessing (Lemmatization, Stopwords removal)
    - Confidence Score
    """)
    st.markdown("---")
    st.text("Built with Streamlit & Scikit-learn")

# --- NLTK Setup ---
@st.cache_resource
def setup_nltk():
    """Download NLTK data if needed."""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except Exception as e:
        st.error(f"NLTK setup error: {e}")
        return False

setup_nltk()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans text: lowercase, remove URLs/emails, keep letters only, remove stopwords, lemmatize.
    """
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\S+|@\S+|\S+@\S+', '', text)  # Remove URLs/Emails
    text = re.sub(r'[^a-z\s]', '', text)                     # Keep letters only
    
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# --- Model Loading & Caching ---
@st.cache_resource
def load_models():
    """Load model and vectorizer with Streamlit caching."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, 'models')
        
        model_path = os.path.join(model_dir, 'fake_news_best_logreg_model.joblib')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            st.error(f"Model files not found in {model_dir}. Please place 'fake_news_best_logreg_model.joblib' and 'tfidf_vectorizer.joblib' there.")
            return None, None
            
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        # Fallback for version mismatch or missing file
        st.error(f"Error loading models: {e}")
        st.warning("Ensure requirements.txt has scikit-learn==1.3.2 if using the pre-trained model, or retain without version pinning if re-training.")
        return None, None

# Load resources
model, vectorizer = load_models()

# --- Main Interface ---

# Input Text Area
news_text = st.text_area("Paste News Content Here:", height=200, placeholder="Type or paste the news article content here...")

# Predict Button
if st.button("Analyze Credibility"):
    if not news_text.strip():
        st.warning("Please enter some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("Model failed to load. Please check the logs.")
    else:
        with st.spinner("Analyzing text patterns..."):
            try:
                # Preprocess
                cleaned_text = clean_text(news_text)
                
                if not cleaned_text:
                    st.warning("The text content seems empty after cleaning (e.g. only numbers or special chars). Please try again with valid text.")
                else:
                    # Vectorize
                    vectorized_text = vectorizer.transform([cleaned_text])
                    
                    # Predict
                    prediction = model.predict(vectorized_text)[0] # 0 (Real) or 1 (Fake)
                    proba = model.predict_proba(vectorized_text)[0] # [prob_0, prob_1]
                    
                    label = "Fake News" if prediction == 1 else "Real News"
                    confidence = proba[prediction] * 100
                    
                    # Display Result
                    st.markdown("---")
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-card fake-news">
                            <h2>🚨 Prediction: {label}</h2>
                            <p>The model is <strong>{confidence:.2f}%</strong> confident this is fake.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-card real-news">
                            <h2>✅ Prediction: {label}</h2>
                            <p>The model is <strong>{confidence:.2f}%</strong> confident this is real.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Debug Info (Optional - Expandable)
                    with st.expander("See processed text"):
                        st.text(cleaned_text)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
