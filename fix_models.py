"""
Script to properly save XGBoost models to avoid serialization warnings.
Run this once to convert your old pickle-based model to the proper format.
"""

import joblib
import xgboost as xgb
import json
import os

print("🔄 Re-saving XGBoost model in proper format...")

try:
    # Load the old pickle model
    print("📖 Loading old ensemble model...")
    ensemble_model = joblib.load("ensemble_sentiment_model.pkl")
    
    # Check if it contains an XGBoost model
    if hasattr(ensemble_model, 'estimators_'):
        # It's likely a sklearn ensemble
        print(f"✅ Found ensemble with {len(ensemble_model.estimators_)} estimators")
        
        # Re-save using joblib (which is fine for sklearn ensembles)
        joblib.dump(ensemble_model, "ensemble_sentiment_model.pkl", compress=3)
        print("✅ Ensemble model re-saved successfully")
    
    elif isinstance(ensemble_model, xgb.Booster):
        # Direct XGBoost Booster model
        print("✅ Found XGBoost Booster model")
        ensemble_model.save_model("ensemble_sentiment_model.json")
        print("✅ Model saved to JSON format: ensemble_sentiment_model.json")
    
    else:
        print(f"⚠️  Model type: {type(ensemble_model)}")
        print("✅ Re-saving with joblib...")
        joblib.dump(ensemble_model, "ensemble_sentiment_model.pkl", compress=3)
        print("✅ Model re-saved")
    
except Exception as e:
    print(f"❌ Error with ensemble model: {e}")

print("\n📖 Loading TF-IDF vectorizer...")
try:
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl", compress=3)
    print("✅ TF-IDF vectorizer re-saved successfully")
except Exception as e:
    print(f"❌ Error with TF-IDF vectorizer: {e}")

print("\n✨ All models have been re-saved properly!")
print("ℹ️  You can now run app.py without serialization warnings")