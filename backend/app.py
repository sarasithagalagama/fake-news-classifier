
from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Initialize Flask
# We serve static files from the ../frontend directory
app = Flask(__name__, static_folder='../frontend', static_url_path='')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Models
try:
    # Path relative to this file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, '../models')
    
    model_path = os.path.join(MODEL_DIR, 'fake_news_best_logreg_model.joblib')
    vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Loading vectorizer from {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise e

# Download NLTK data (if not present)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Cleans text: lowercase, remove URLs/emails, keep letters only, remove stopwords, lemmatize.
    Must match the preprocessing used during training.
    """
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\S+|@\S+|\S+@\S+', '', text)  # Remove URLs/Emails
    text = re.sub(r'[^a-z\s]', '', text)                     # Keep letters only
    
    # Tokenize & Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def index():
    """Serve the frontend HTML."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    """Serve other static files (css, js)."""
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocessing
        cleaned_text = clean_text(text)
        
        # Vectorize
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(vectorized_text)[0] # 0 (Real) or 1 (Fake)
        proba = model.predict_proba(vectorized_text)[0] # [prob_0, prob_1]
        
        label = "Fake" if prediction == 1 else "Real"
        confidence = proba[prediction]
        
        return jsonify({
            'prediction': label,
            'confidence': float(confidence),
            'cleaned_text': cleaned_text  # Optional: for debugging
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use PORT env variable if available (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
