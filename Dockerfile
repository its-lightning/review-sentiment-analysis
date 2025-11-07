# Stage 1: Base image with Python
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords punkt punkt_tab wordnet

# Copy application files
COPY app.py .
COPY fix_models.py .
COPY ensemble_sentiment_model.pkl .
COPY tfidf_vectorizer.pkl .
COPY ensemble_sentiment_model_final.pkl .
COPY tfidf_vectorizer_final.pkl .
COPY templates/ ./templates/
COPY static/ ./static/

# Expose port 5050
EXPOSE 5050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5050/health || exit 1

# Run the Flask application
CMD ["python", "app.py"]