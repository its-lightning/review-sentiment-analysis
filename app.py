from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import joblib
import numpy as np
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
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=60,
    ping_interval=25,
)

CONFIDENCE_THRESHOLD = 0.82  # Only keep predictions above this confidence

class TqdmTfidfVectorizer(TfidfVectorizer):
    pass

try:
    model_bundle = joblib.load("ensemble_sentiment_model_final.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer_final.pkl")

    if isinstance(model_bundle, dict):
        ensemble_model = model_bundle.get("meta_learner")
    else:
        ensemble_model = model_bundle

    logger.info("Loaded ensemble sentiment model and vectorizer successfully.")
except Exception as e:
    logger.critical(f"Model loading failed: {e}")
    raise

for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words and len(tok) > 2]
    return " ".join(cleaned)


def is_valid_youtube_url(url):
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=",
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/",
    ]
    return any(re.search(pattern, url) for pattern in patterns)


def predict_sentiment(text):
    processed = preprocess_text(text)
    if not processed:
        return None

    X = tfidf_vectorizer.transform([processed])

    if hasattr(ensemble_model, "predict_proba"):
        prob = ensemble_model.predict_proba(X)[0]
        pred = np.argmax(prob)
        confidence = prob[pred]
    else:
        pred = ensemble_model.predict(X)[0]
        confidence = 1.0

    if confidence < CONFIDENCE_THRESHOLD:
        return None  # skip uncertain predictions

    sentiment = "Positive" if pred == 1 else "Negative"
    rating = round(confidence * 5.0, 2)
    return {"comment": text, "sentiment": sentiment, "confidence": round(confidence, 3), "rating": rating}


@socketio.on("process_video")
def process_video(data):
    url = data.get("url", "").strip()
    if not url:
        emit("error", {"message": "No URL provided"})
        return
    if not is_valid_youtube_url(url):
        emit("error", {"message": "Invalid YouTube URL"})
        return

    logger.info(f"Processing video: {url}")
    emit("status", {"message": "Downloading comments..."})
    emit("progress", {"stage": "download", "value": 0})

    comment_queue = Queue()
    result_queue = Queue()

    def downloader():
        try:
            downloader = YoutubeCommentDownloader()
            count = 0
            for c in downloader.get_comments_from_url(url):
                text = c.get("text", "")
                if text:
                    comment_queue.put(text)
                    count += 1
                    if count % 50 == 0:
                        socketio.emit("progress", {"stage": "download", "value": count})
            comment_queue.put(None)
            result_queue.put(("total_comments", count))
            socketio.emit("status", {"message": f"Downloaded {count} comments. Analyzing..."})
        except Exception as e:
            logger.error(f"Download error: {e}")
            result_queue.put(("error", str(e)))

    def analyzer():
        results = []
        pos_count, neg_count = 0, 0
        total = 0

        try:
            while True:
                text = comment_queue.get()
                if text is None:
                    break
                result = predict_sentiment(text)
                if not result:
                    continue  # skip low-confidence results

                results.append(result)
                total += 1
                if result["sentiment"] == "Positive":
                    pos_count += 1
                else:
                    neg_count += 1

                if total % 50 == 0:
                    socketio.emit("progress", {"stage": "analysis", "value": total})

            avg_conf = np.mean([r["confidence"] for r in results]) if results else 0.0
            result_queue.put(("done", {
                "total": total,
                "valid_count": total,
                "average_rating": round(avg_conf * 5, 2),
                "results": results,
                "sentiment_counts": {"Positive": pos_count, "Negative": neg_count},
            }))
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            result_queue.put(("error", str(e)))

    Thread(target=downloader, daemon=True).start()
    Thread(target=analyzer, daemon=True).start()

    try:
        while True:
            msg_type, msg_data = result_queue.get(timeout=600)
            if msg_type == "error":
                emit("error", {"message": msg_data})
                return
            elif msg_type == "done":
                emit("done", msg_data)
                logger.info(f"Analysis done: {msg_data['sentiment_counts']} (threshold={CONFIDENCE_THRESHOLD})")
                break
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        emit("error", {"message": str(e)})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    logger.info(f"Starting 2-Class Ensemble Sentiment Analyzer (Confidence Threshold={CONFIDENCE_THRESHOLD})")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False, allow_unsafe_werkzeug=True)
