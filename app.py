from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import joblib
import numpy as np
import pandas as pd
import re
from youtube_comment_downloader import YoutubeCommentDownloader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Flask setup ---
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:5050", "http://localhost:3000"])

# --- Load ML models ---
try:
    ensemble_model = joblib.load("ensemble_sentiment_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    logger.info("Models loaded successfully")
except FileNotFoundError as e:
    logger.critical(f"Model files not found: {e}")
    raise

# --- NLTK setup ---
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
except Exception as e:
    logger.warning(f"NLTK download issue: {e}")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# --- Helper functions ---
def is_valid_youtube_url(url):
    """Validate if URL is a YouTube URL"""
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/'
    ]
    return any(re.search(pattern, url) for pattern in patterns)


def preprocess_text(text):
    """Preprocess and clean text"""
    if not text or (isinstance(text, float) and pd.isna(text)):
        return ""
    
    text = str(text).strip().lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    
    return " ".join(tokens)


def convert_to_rating(sentiment_probs):
    """Convert sentiment probabilities to 1-5 rating scale"""
    ratings = []
    for probs in sentiment_probs:
        rating = (probs[0] * 1.5) + (probs[1] * 3.0) + (probs[2] * 4.5)
        rating = max(1.0, min(5.0, rating))
        ratings.append(rating)
    return np.array(ratings)


# --- SocketIO route for processing ---
@socketio.on("process_video")
def handle_process_video(data):
    """Handle video processing request"""
    video_url = data.get("url", "").strip()
    
    if not video_url:
        emit("error", {"message": "No URL provided"})
        logger.warning("Empty URL provided")
        return
    
    if not is_valid_youtube_url(video_url):
        emit("error", {"message": "Invalid YouTube URL. Please enter a valid youtube.com or youtu.be link"})
        logger.warning(f"Invalid URL format: {video_url}")
        return

    emit("status", {"message": "🔍 Downloading comments..."})
    logger.info(f"Starting to process video: {video_url}")
    
    try:
        downloader = YoutubeCommentDownloader()
        comments_data = downloader.get_comments_from_url(video_url)
        comments = [c.get("text", "") for c in comments_data if c.get("text")]
        
    except ValueError as e:
        emit("error", {"message": "Invalid YouTube URL or video not found"})
        logger.error(f"ValueError: {e}")
        return
    except ConnectionError:
        emit("error", {"message": "Network error. Please check your connection"})
        logger.error("Connection error while downloading comments")
        return
    except Exception as e:
        emit("error", {"message": f"Failed to fetch comments: {str(e)}"})
        logger.error(f"Unexpected error: {e}")
        return

    if not comments:
        emit("error", {"message": "No comments found on this video or comments are disabled"})
        logger.info(f"No comments found for video: {video_url}")
        return

    emit("status", {"message": f"✅ Downloaded {len(comments)} comments"})
    logger.info(f"Downloaded {len(comments)} comments")
    
    total = len(comments)
    processed = []

    emit("status", {"message": "📝 Preprocessing text..."})
    
    for i, c in enumerate(comments):
        processed.append(preprocess_text(c))
        
        # Emit progress every 50 comments or at the end
        if (i + 1) % 50 == 0 or i == total - 1:
            progress = int(((i + 1) / total) * 100)
            socketio.emit("progress", {"progress": progress})

    emit("status", {"message": "🤖 Analyzing sentiments..."})
    logger.info("Starting sentiment analysis")
    
    try:
        X = tfidf_vectorizer.transform(processed)
        probabilities = ensemble_model.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        ratings = convert_to_rating(probabilities)
    except Exception as e:
        emit("error", {"message": "Error during sentiment analysis"})
        logger.error(f"Sentiment analysis error: {e}")
        return

    threshold = 0.3
    valid = [
        (c, p, r, probabilities[i][p])
        for i, (c, p, r) in enumerate(zip(comments, predictions, ratings))
        if probabilities[i][p] >= threshold
    ]

    labels = ["Negative", "Neutral", "Positive"]
    
    if not valid:
        logger.info("No comments met confidence threshold")
        emit("done", {
            "total": total,
            "valid_count": 0,
            "average_rating": None,
            "results": [],
            "sentiment_counts": {"Negative": 0, "Neutral": 0, "Positive": 0}
        })
        return

    avg_rating = np.mean([v[2] for v in valid])
    
    # Count sentiments
    sentiment_counts = {label: 0 for label in labels}
    for v in valid:
        sentiment_counts[labels[v[1]]] += 1
    
    results = [
        {
            "comment": v[0],
            "sentiment": labels[v[1]],
            "confidence": round(v[3], 3),
            "rating": round(v[2], 2),
        }
        for v in valid
    ]

    logger.info(f"Analysis complete: {len(valid)} valid comments, avg rating: {avg_rating:.2f}")
    
    emit("done", {
        "total": total,
        "valid_count": len(valid),
        "average_rating": round(avg_rating, 2),
        "results": results,
        "sentiment_counts": sentiment_counts
    })


@app.route("/")
def index():
    """Render main page"""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok"}, 200


if __name__ == "__main__":
    logger.info("Starting YouTube Sentiment Analyzer")
    socketio.run(app, host="0.0.0.0", port=5050, debug=True)