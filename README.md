
# Fake News Classifier Web App

This is a Flask-based web application that uses a trained Logistic Regression model to classify news articles as "Real" or "Fake".

## Project Structure
- `backend/`: Contains the Flask application (`app.py`).
- `frontend/`: Contains the HTML, CSS, and JS files for the UI.
- `models/`: Contains the trained `.joblib` model and vectorizer.
- `requirements.txt`: Python dependencies.

## Local Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python backend/app.py
   ```
   
3. Open http://localhost:5000 in your browser.

## Deployment on Render.com

1. Push this repository to GitHub/GitLab.
2. Log in to [Render.com](https://render.com).
3. Click "New +" -> "Web Service".
4. Connect your repository.
5. In the settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn backend.app:app`
6. Click "Create Web Service".

The app will download NLTK data on startup and serve the frontend.
