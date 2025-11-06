from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import joblib
import numpy as np
import re
import logging
import warnings
from queue import Queue
from threading import Thread
from youtube_comment_downloader import YoutubeCommentDownloader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'youtube-sentiment-analyzer-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ===========================================
# FIX: Define TqdmTfidfVectorizer before loading
# ===========================================
from sklearn.feature_extraction.text import TfidfVectorizer

class TqdmTfidfVectorizer(TfidfVectorizer):
    """Custom TfidfVectorizer class for compatibility with saved models."""
    pass

# ===========================================
# LOAD ENSEMBLE MODEL PROPERLY
# ===========================================
try:
    logger.info("Loading models...")
    ensemble_bundle = joblib.load("ensemble_sentiment_model_final.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer_final.pkl")
    
    # Unpack the ensemble bundle
    meta_learner = ensemble_bundle['meta_learner']
    base_models = ensemble_bundle['base_models']
    model_names = ensemble_bundle['model_names']
    label_offset = ensemble_bundle.get('label_offset', 1)  # Base models predict [1,2], subtract to get [0,1]
    metadata = ensemble_bundle.get('metadata', {})
    
    logger.info("✅ Models loaded successfully")
    logger.info(f"   Base models: {', '.join(model_names)}")
    logger.info(f"   Meta-learner: {type(meta_learner).__name__}")
    logger.info(f"   Label offset: {label_offset} (base models predict [1,2], converted to [0,1])")
    logger.info(f"   Ensemble accuracy: {metadata.get('accuracy', 'N/A')}")
    
except FileNotFoundError as e:
    logger.critical(f"❌ Model files not found: {e}")
    logger.critical("   Please ensure 'ensemble_sentiment_model_final.pkl' and 'tfidf_vectorizer_final.pkl' exist")
    raise SystemExit(1)
except Exception as e:
    logger.critical(f"❌ Error loading models: {e}")
    import traceback
    logger.critical(traceback.format_exc())
    raise SystemExit(1)

# ===========================================
# NLTK SETUP
# ===========================================
for resource in ['stopwords', 'punkt', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else 'tokenizers/punkt')
    except LookupError:
        logger.info(f"Downloading NLTK resource: {resource}")
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ===========================================
# CONFIGURATION - ADJUST THESE VALUES
# ===========================================
CONFIDENCE_THRESHOLD = 0.55  # Lower threshold to capture more predictions (default: 0.60)
MAX_COMMENTS = 5000  # Maximum comments to download
BATCH_UPDATE_SIZE = 25  # Update UI every N comments

# ===========================================
# HELPER FUNCTIONS
# ===========================================
def is_valid_youtube_url(url):
    """Check if URL is valid YouTube link and extract video ID."""
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([\w\-]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([\w\-]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([\w\-]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return True, match.group(1)
    return False, None


def preprocess_text(text):
    """
    Clean and preprocess text EXACTLY like training script.
    CRITICAL: Must match training preprocessing pipeline.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags and URLs
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stopwords
    cleaned_tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 2
    ]
    
    return " ".join(cleaned_tokens)


def predict_sentiment(text):
    """
    Predict sentiment using the ensemble model.
    FIXED: Properly handle binary classification with label conversion.
    Returns: (label, confidence, probs, debug_info)
    """
    try:
        # Preprocess text
        processed = preprocess_text(text)
        if not processed:
            return "Neutral", 0.5, [0.5, 0.5], {"error": "empty_after_preprocessing"}
        
        # Vectorize using TF-IDF
        X_tfidf = tfidf_vectorizer.transform([processed])
        
        # Get base model predictions
        # CRITICAL: Base models were trained on labels [1,2], so they predict [1,2]
        # We need to convert to [0,1] by subtracting label_offset
        base_predictions = []
        base_predictions_original = []
        for model_name, model in base_models:
            pred_original = model.predict(X_tfidf)[0]  # Predicts 1 or 2
            pred_binary = pred_original - label_offset  # Convert to 0 or 1
            base_predictions.append(pred_binary)
            base_predictions_original.append(pred_original)
        
        # Reshape for meta-learner input: (1, n_base_models)
        meta_features = np.array(base_predictions).reshape(1, -1)
        
        # Get final prediction from meta-learner
        final_pred = meta_learner.predict(meta_features)[0]
        
        # Validate prediction (must be 0 or 1)
        if final_pred not in [0, 1]:
            logger.warning(f"Invalid prediction {final_pred}, defaulting to 0")
            final_pred = 0
        
        # Get probabilities
        if hasattr(meta_learner, 'predict_proba'):
            probs = meta_learner.predict_proba(meta_features)[0]
        else:
            # Fallback: create one-hot encoding
            probs = np.zeros(2)
            probs[int(final_pred)] = 1.0
        
        # Map to sentiment label
        label = "Positive" if final_pred == 1 else "Negative"
        confidence = float(probs[int(final_pred)])
        
        # Debug info
        debug_info = {
            "base_preds_original": base_predictions_original,
            "base_preds_binary": base_predictions,
            "meta_features": meta_features.tolist(),
            "final_pred": int(final_pred),
            "probs": probs.tolist()
        }
        
        return label, confidence, probs.tolist(), debug_info
        
    except Exception as e:
        logger.error(f"⚠️ Prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return "Neutral", 0.5, [0.5, 0.5], {"error": str(e)}


def convert_to_rating(confidence, sentiment):
    """
    Convert binary sentiment to 1-5 star rating scale.
    - Negative sentiment: 1.0 - 2.5 stars
    - Positive sentiment: 2.5 - 5.0 stars
    """
    if sentiment == "Positive":
        # Map positive confidence (0.5-1.0) to rating (2.5-5.0)
        return 2.5 + (confidence * 2.5)
    else:
        # Map negative confidence (0.5-1.0) to rating (2.5-1.0)
        return 2.5 - ((confidence - 0.5) * 3.0)


# ===========================================
# SOCKET HANDLER
# ===========================================
@socketio.on("process_video")
def process_video(data):
    """Main handler for video comment analysis with detailed debugging."""
    video_url = data.get("url", "").strip()
    valid, video_id = is_valid_youtube_url(video_url)

    if not valid:
        emit("error", {"message": "Invalid YouTube URL. Please use youtube.com or youtu.be links."})
        return

    emit("status", {"message": "Starting comment download..."})
    logger.info(f"🎥 Processing video: {video_id}")

    comment_queue = Queue()
    result_queue = Queue()

    # ----------------------------
    # DOWNLOADER THREAD
    # ----------------------------
    def downloader_worker():
        try:
            downloader = YoutubeCommentDownloader()
            count = 0
            logger.info("⬇️ Starting comment download...")
            
            for comment in downloader.get_comments_from_url(video_url, sort_by=0):
                text = comment.get("text", "")
                if text and len(text.strip()) > 5:  # Filter very short comments
                    comment_queue.put(("comment", text))
                    count += 1
                    
                    if count % BATCH_UPDATE_SIZE == 0:
                        logger.info(f"📥 Downloaded {count} comments")
                        socketio.emit("progress", {
                            "stage": "download", 
                            "value": count
                        })
                
                # Limit to prevent excessive processing
                if count >= MAX_COMMENTS:
                    logger.info(f"⚠️ Reached {MAX_COMMENTS} comment limit")
                    break
            
            if count == 0:
                logger.warning("⚠️ No comments found")
                comment_queue.put(("error", "No comments found for this video. The video may have comments disabled."))
            else:
                comment_queue.put(("done", count))
                logger.info(f"✅ Download complete: {count} comments")
                
        except Exception as e:
            logger.error(f"❌ Downloader failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            comment_queue.put(("error", f"Failed to download comments: {str(e)}"))

    # ----------------------------
    # ANALYZER THREAD WITH DEBUGGING
    # ----------------------------
    def analyzer_worker():
        valid_results = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        sentiment_distribution = {"Positive": 0, "Negative": 0}  # For display (excludes Neutral)
        total_downloaded = 0
        analyzed_count = 0
        skipped_neutral = 0
        skipped_low_confidence = 0
        
        # Debug: Track first 10 predictions for analysis
        debug_predictions = []
        
        try:
            while True:
                msg_type, msg_data = comment_queue.get()
                
                if msg_type == "error":
                    result_queue.put(("error", msg_data))
                    break
                    
                elif msg_type == "done":
                    total_downloaded = msg_data
                    logger.info(f"🔍 Analysis complete: {analyzed_count}/{total_downloaded} comments analyzed")
                    logger.info(f"   Sentiment counts: {sentiment_counts}")
                    logger.info(f"   Skipped neutral: {skipped_neutral}")
                    logger.info(f"   Skipped low confidence: {skipped_low_confidence}")
                    logger.info(f"   Valid results: {len(valid_results)}")
                    
                    # Log debug predictions
                    if debug_predictions:
                        logger.info(f"\n=== First 10 Predictions (Debug) ===")
                        for i, debug in enumerate(debug_predictions[:10], 1):
                            logger.info(f"{i}. Comment: '{debug['comment'][:50]}...'")
                            logger.info(f"   Sentiment: {debug['sentiment']} (confidence: {debug['confidence']:.3f})")
                            logger.info(f"   Base preds (original): {debug['base_preds_original']}")
                            logger.info(f"   Base preds (binary): {debug['base_preds_binary']}")
                            logger.info(f"   Probabilities: {debug['probs']}")
                    
                    result_queue.put(("done", {
                        "results": valid_results,
                        "sentiment_counts": sentiment_distribution,
                        "total_count": total_downloaded,
                        "analyzed_count": analyzed_count,
                        "stats": {
                            "all_sentiments": sentiment_counts,
                            "skipped_neutral": skipped_neutral,
                            "skipped_low_confidence": skipped_low_confidence
                        }
                    }))
                    break
                    
                elif msg_type == "comment":
                    raw_comment = msg_data
                    
                    # Predict sentiment
                    label, confidence, probs, debug_info = predict_sentiment(raw_comment)
                    
                    analyzed_count += 1
                    sentiment_counts[label] += 1
                    
                    # Store debug info for first 10 predictions
                    if len(debug_predictions) < 10:
                        debug_predictions.append({
                            "comment": raw_comment,
                            "sentiment": label,
                            "confidence": confidence,
                            "probs": probs,
                            "base_preds_original": debug_info.get("base_preds_original", []),
                            "base_preds_binary": debug_info.get("base_preds_binary", [])
                        })
                    
                    # Skip neutral predictions
                    if label == "Neutral":
                        skipped_neutral += 1
                        continue
                    
                    # Apply confidence threshold
                    if confidence >= CONFIDENCE_THRESHOLD:
                        rating = convert_to_rating(confidence, label)
                        
                        valid_results.append({
                            "comment": raw_comment[:200],  # Truncate very long comments
                            "sentiment": label,
                            "confidence": round(confidence, 3),
                            "rating": round(rating, 2)
                        })
                        sentiment_distribution[label] += 1
                    else:
                        skipped_low_confidence += 1
                    
                    # Emit progress updates
                    if analyzed_count % BATCH_UPDATE_SIZE == 0:
                        socketio.emit("progress", {
                            "stage": "analysis",
                            "value": analyzed_count
                        })
                        
        except Exception as e:
            logger.error(f"❌ Analyzer error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result_queue.put(("error", f"Analysis failed: {str(e)}"))

    # Start threads
    Thread(target=downloader_worker, daemon=True).start()
    Thread(target=analyzer_worker, daemon=True).start()

    # Wait for results
    try:
        result_type, result_data = result_queue.get(timeout=600)  # 10 minute timeout
        
        if result_type == "error":
            emit("error", {"message": result_data})
            return
        
        # Calculate statistics
        ratings = [r["rating"] for r in result_data["results"]]
        avg_rating = np.mean(ratings) if ratings else None
        
        emit("done", {
            "total": result_data["total_count"],
            "valid_count": len(result_data["results"]),
            "average_rating": round(avg_rating, 2) if avg_rating else None,
            "results": result_data["results"],
            "sentiment_counts": result_data["sentiment_counts"],
            "stats": result_data.get("stats", {})
        })
        
        logger.info(f"✅ Processing complete — {result_data['total_count']} comments, {len(result_data['results'])} analyzed")
        logger.info(f"   Sentiment breakdown: {result_data['sentiment_counts']}")
        logger.info(f"   Average rating: {avg_rating:.2f}" if avg_rating else "   No valid ratings")
        
    except Exception as e:
        logger.error(f"❌ Processing timeout/error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        emit("error", {"message": f"Processing failed: {str(e)}"})


# ===========================================
# ROUTES
# ===========================================
@app.route("/")
def index():
    """Serve main application page."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "model_type": "ensemble_stacking",
        "base_models": model_names,
        "num_base_models": len(base_models),
        "meta_learner": type(meta_learner).__name__,
        "accuracy": metadata.get('accuracy'),
        "f1_score": metadata.get('f1_score'),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "max_comments": MAX_COMMENTS
    }), 200


@app.route("/api/test", methods=["POST"])
def test_prediction():
    """Test endpoint for debugging predictions."""
    from flask import request
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    label, confidence, probs, debug_info = predict_sentiment(text)
    rating = convert_to_rating(confidence, label)
    
    return jsonify({
        "text": text,
        "sentiment": label,
        "confidence": confidence,
        "probabilities": probs,
        "rating": rating,
        "debug": debug_info,
        "threshold": CONFIDENCE_THRESHOLD,
        "passes_threshold": confidence >= CONFIDENCE_THRESHOLD
    }), 200


# ===========================================
# MAIN
# ===========================================
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("🚀 YouTube Sentiment Analyzer Starting...")
    logger.info("="*60)
    logger.info(f"   Ensemble: {', '.join(model_names)}")
    logger.info(f"   Meta-learner: {type(meta_learner).__name__}")
    logger.info(f"   Number of base models: {len(base_models)}")
    logger.info(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.info(f"   Max comments: {MAX_COMMENTS}")
    if metadata:
        logger.info(f"   Model accuracy: {metadata.get('accuracy', 'N/A')}")
    logger.info("="*60)
    
    socketio.run(app, host="0.0.0.0", port=5050, debug=False, allow_unsafe_werkzeug=True)