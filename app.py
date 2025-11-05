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
import warnings
from queue import Queue
from threading import Thread

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Flask setup ---
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e6,
    engineio_logger=False,
    socketio_logger=False
)

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
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception as e:
        logger.warning(f"Could not download stopwords: {e}")

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception as e:
        logger.warning(f"Could not download punkt_tab: {e}")

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    try:
        nltk.download("wordnet", quiet=True)
    except Exception as e:
        logger.warning(f"Could not download wordnet: {e}")

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
    """Convert sentiment probabilities to 0-5 rating scale"""
    ratings = []
    for probs in sentiment_probs:
        # Get the index and probability of the predicted sentiment
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        # Map based on which class was predicted:
        # Class 0 = Negative -> map to 0-2
        # Class 1 = Neutral -> map to 2-3
        # Class 2 = Positive -> map to 3-5
        if pred_idx == 0:  # Negative
            rating = confidence * 2.0
        elif pred_idx == 1:  # Neutral
            rating = 2.0 + (confidence * 1.0)
        else:  # Positive (pred_idx == 2)
            rating = 3.0 + (confidence * 2.0)
        
        rating = max(0.0, min(5.0, rating))
        ratings.append(rating)
    return np.array(ratings)


def map_score_to_sentiment(score):
    """Map model output score (0-5) to sentiment category
    0-2: Negative
    2-3: Neutral
    3-5: Positive
    """
    if score < 2:
        return "Negative"
    elif score < 3:
        return "Neutral"
    else:
        return "Positive"


# --- SocketIO route for processing ---
@socketio.on("process_video")
def handle_process_video(data):
    """Handle video processing request with async analysis"""
    video_url = data.get("url", "").strip()
    
    logger.info(f"Received request for video: {video_url}")
    
    if not video_url:
        emit("error", {"message": "No URL provided"})
        logger.warning("Empty URL provided")
        return
    
    if not is_valid_youtube_url(video_url):
        emit("error", {"message": "Invalid YouTube URL. Please enter a valid youtube.com or youtu.be link"})
        logger.warning(f"Invalid URL format: {video_url}")
        return

    emit("status", {"message": "Downloading and analyzing comments..."})
    emit("progress", {"stage": "download", "value": 0})
    logger.info(f"Starting async processing for video: {video_url}")
    
    # Queues for thread communication
    comment_queue = Queue()
    results_queue = Queue()
    
    def downloader_worker():
        """Download comments and put them in queue"""
        try:
            logger.info("Downloader worker started")
            downloader = YoutubeCommentDownloader()
            comment_count = 0
            
            for comment in downloader.get_comments_from_url(video_url):
                text = comment.get("text", "")
                if text:
                    comment_queue.put(("comment", text))
                    comment_count += 1
                    
                    # Emit download progress every 50 comments
                    if comment_count % 50 == 0:
                        logger.info(f"Downloaded {comment_count} comments")
                        socketio.emit("progress", {"stage": "download", "value": comment_count})
            
            comment_queue.put(("total_comments", comment_count))
            comment_queue.put(("done", None))
            logger.info(f"Download complete: {comment_count} total comments")
            socketio.emit("status", {"message": f"✅ Downloaded {comment_count} comments. Analyzing..."})
            socketio.emit("progress", {"stage": "download", "value": comment_count})
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            comment_queue.put(("error", str(e)))
    
    def analyzer_worker():
        """Analyze comments as they arrive"""
        threshold = 0.50
        valid_results = []
        total_comments = 0
        total_to_analyze = 0
        sentiment_counts = {"Negative": 0, "Positive": 0}
        
        try:
            logger.info("Analyzer worker started")
            while True:
                msg_type, msg_data = comment_queue.get()
                
                if msg_type == "error":
                    logger.error(f"Error from downloader: {msg_data}")
                    results_queue.put(("error", msg_data))
                    break
                
                elif msg_type == "total_comments":
                    total_to_analyze = msg_data
                    logger.info(f"Will analyze {total_to_analyze} comments")
                
                elif msg_type == "done":
                    logger.info(f"Analysis complete: {len(valid_results)} valid out of {total_comments} comments")
                    results_queue.put(("done", {
                        "valid_count": len(valid_results),
                        "total_count": total_comments,
                        "sentiment_counts": sentiment_counts,
                        "results": valid_results
                    }))
                    break
                
                elif msg_type == "comment":
                    total_comments += 1
                    text = preprocess_text(msg_data)
                    
                    # Analyze single comment
                    try:
                        X = tfidf_vectorizer.transform([text])
                        prob = ensemble_model.predict_proba(X)[0]
                        pred = np.argmax(prob)
                        rating = convert_to_rating([prob])[0]
                        confidence = prob[pred]
                        
                        # Map to sentiment based on rating score
                        sentiment_label = map_score_to_sentiment(rating)
                        
                        # If meets threshold AND not neutral, add to results
                        if confidence >= threshold and sentiment_label != "Neutral":
                            valid_results.append({
                                "comment": msg_data,
                                "sentiment": sentiment_label,
                                "confidence": round(confidence, 3),
                                "rating": round(rating, 2)
                            })
                            sentiment_counts[sentiment_label] += 1
                        
                        # Emit progress every 50 comments
                        if total_comments % 50 == 0:
                            logger.info(f"Analyzed {total_comments}/{total_to_analyze} comments | Valid: {len(valid_results)}")
                            socketio.emit("progress", {"stage": "analysis", "value": total_comments})
                        
                    except Exception as e:
                        logger.error(f"Analysis error for comment #{total_comments}: {e}")
        
        except Exception as e:
            logger.error(f"Analyzer worker error: {e}")
            results_queue.put(("error", str(e)))
    
    # Start both threads
    logger.info("Starting download and analysis threads")
    downloader_thread = Thread(target=downloader_worker, daemon=True)
    analyzer_thread = Thread(target=analyzer_worker, daemon=True)
    
    downloader_thread.start()
    analyzer_thread.start()
    
    # Wait for results
    try:
        result_type, result_data = results_queue.get(timeout=600)
        
        if result_type == "error":
            logger.error(f"Processing error: {result_data}")
            emit("error", {"message": f"Processing error: {result_data}"})
        else:
            emit("progress", {"stage": "analysis", "value": result_data['total_count']})
            
            avg_rating = None
            if result_data["valid_count"] > 0:
                avg_rating = np.mean([r["rating"] for r in result_data["results"]])
            
            logger.info(f"Processing complete: {result_data['valid_count']} valid out of {result_data['total_count']} total")
            logger.info(f"Sentiment breakdown: {result_data['sentiment_counts']}")
            avg_rating_display = round(avg_rating, 2) if avg_rating else "N/A"
            logger.info(f"Average rating: {avg_rating_display}")
            
            emit("done", {
                "total": result_data["total_count"],
                "valid_count": result_data["valid_count"],
                "average_rating": round(avg_rating, 2) if avg_rating else None,
                "results": result_data["results"],
                "sentiment_counts": result_data["sentiment_counts"]
            })
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        emit("error", {"message": f"Processing failed: {str(e)}"})


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
    socketio.run(app, host="0.0.0.0", port=5050, debug=False, allow_unsafe_werkzeug=True)