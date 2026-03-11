import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

# ─────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────

app = FastAPI(
    title       = "Spam & Fraud Detection API",
    description = "REST API for spam classification and behavioral anomaly detection",
    version     = "1.0.0"
)


# ─────────────────────────────────────────
# Load Models at Startup
# ─────────────────────────────────────────

print("Loading models...")

# SMS Spam Model
sms_model      = joblib.load("models/sms_spam_model.pkl")
sms_vectorizer = joblib.load("models/sms_spam_vectorizer.pkl")

# Email Spam Model
email_model    = joblib.load("models/email_spam_model.pkl")

# Behavior Model
behavior_model  = joblib.load("models/behavior_model.pkl")
behavior_scaler = joblib.load("models/behavior_scaler.pkl")

# Metrics (from training results)
sms_results   = joblib.load("models/sms_training_results.pkl")
email_results = joblib.load("models/email_training_results.pkl")

print("All models loaded successfully!")


# ─────────────────────────────────────────
# Request & Response Schemas
# ─────────────────────────────────────────

class SMSRequest(BaseModel):
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Congratulations! You won a free prize. Click here now!"
            }
        }


class EmailRequest(BaseModel):
    num_links:    int
    num_words:    int
    has_offer:    int
    sender_score: float
    all_caps:     int

    class Config:
        json_schema_extra = {
            "example": {
                "num_links":    5,
                "num_words":    120,
                "has_offer":    1,
                "sender_score": 0.3,
                "all_caps":     1
            }
        }


class BehaviorRequest(BaseModel):
    user_id:                  Optional[int] = None
    messages_per_minute:      float
    unique_recipients:        int
    avg_time_between_msgs:    float
    repeated_message_ratio:   float
    night_activity_ratio:     float
    failed_message_ratio:     float

    class Config:
        json_schema_extra = {
            "example": {
                "user_id":                  1,
                "messages_per_minute":      350.0,
                "unique_recipients":        800,
                "avg_time_between_msgs":    0.5,
                "repeated_message_ratio":   0.85,
                "night_activity_ratio":     0.75,
                "failed_message_ratio":     0.30
            }
        }


# ─────────────────────────────────────────
# Helper — Risk Score
# ─────────────────────────────────────────

def compute_risk_score(raw_score: float) -> int:
    """
    Convert Isolation Forest raw score to 0-100 risk score.
    More negative score = higher risk.
    """
    # Clip and normalize to 0-100
    # Typical range is -0.5 to 0.5
    normalized = (0.5 - raw_score) / 1.0
    risk       = int(np.clip(normalized * 100, 0, 100))
    return risk


# ─────────────────────────────────────────
# Helper — Top Feature
# ─────────────────────────────────────────

FEATURE_COLS = [
    'messages_per_minute',
    'unique_recipients',
    'avg_time_between_msgs',
    'repeated_message_ratio',
    'night_activity_ratio',
    'failed_message_ratio'
]

def get_top_feature(scaled_row: np.ndarray) -> str:
    """Return feature with highest absolute deviation"""
    top_idx = np.argmax(np.abs(scaled_row))
    return FEATURE_COLS[top_idx]


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Spam & Fraud Detection API",
        "version": "1.0.0",
        "endpoints": [
            "POST /predict/spam/sms",
            "POST /predict/spam/email",
            "POST /predict/behavior",
            "GET  /metrics"
        ]
    }


# ── POST /predict/spam/sms ──────────────

@app.post("/predict/spam/sms")
def predict_sms_spam(request: SMSRequest):
    """
    Classify an SMS message as spam or ham.

    Args:
        message : raw SMS text

    Returns:
        label       : spam or ham
        probability : confidence score 0-1
        is_spam     : True or False
    """

    if not request.message.strip():
        raise HTTPException(
            status_code = 400,
            detail      = "Message cannot be empty"
        )

    # Vectorize and predict
    X        = sms_vectorizer.transform([request.message])
    pred     = sms_model.predict(X)[0]
    prob     = sms_model.predict_proba(X)[0]

    is_spam  = bool(pred == 1)
    spam_prob = round(float(prob[1]), 4)

    return {
        "input":       request.message,
        "label":       "spam" if is_spam else "ham",
        "is_spam":     is_spam,
        "probability": spam_prob,
        "model":       "SVM"
    }


# ── POST /predict/spam/email ────────────

@app.post("/predict/spam/email")
def predict_email_spam(request: EmailRequest):
    """
    Classify an email as spam or ham.

    Args:
        num_links, num_words, has_offer,
        sender_score, all_caps

    Returns:
        label       : spam or ham
        probability : confidence score 0-1
        is_spam     : True or False
    """

    features = np.array([[
        request.num_links,
        request.num_words,
        request.has_offer,
        request.sender_score,
        request.all_caps
    ]])

    pred     = email_model.predict(features)[0]
    prob     = email_model.predict_proba(features)[0]

    is_spam  = bool(pred == 1)
    spam_prob = round(float(prob[1]), 4)

    return {
        "input":       request.dict(),
        "label":       "spam" if is_spam else "ham",
        "is_spam":     is_spam,
        "probability": spam_prob,
        "model":       "Logistic Regression"
    }


# ── POST /predict/behavior ──────────────

@app.post("/predict/behavior")
def predict_behavior(request: BehaviorRequest):
    """
    Detect anomalous user behavior.

    Args:
        6 behavioral features

    Returns:
        risk_score  : 0-100
        anomaly     : True or False
        top_feature : most suspicious feature
    """

    features = np.array([[
        request.messages_per_minute,
        request.unique_recipients,
        request.avg_time_between_msgs,
        request.repeated_message_ratio,
        request.night_activity_ratio,
        request.failed_message_ratio
    ]])

    # Normalize
    scaled      = behavior_scaler.transform(features)

    # Predict
    pred        = behavior_model.predict(scaled)[0]
    raw_score   = behavior_model.score_samples(scaled)[0]

    is_anomaly  = bool(pred == -1)
    risk_score  = compute_risk_score(raw_score)
    top_feature = get_top_feature(scaled[0])

    return {
        "user_id":     request.user_id,
        "risk_score":  risk_score,
        "anomaly":     is_anomaly,
        "top_feature": top_feature,
        "model":       "Isolation Forest"
    }


# ── GET /metrics ────────────────────────

@app.get("/metrics")
def get_metrics():
    """
    Return F1, Precision, Recall and AUC
    for both SMS and Email spam models.
    """

    sms_best   = sms_results['best_name']
    email_best = email_results['best_name']

    sms_m      = sms_results['results'][sms_best]
    email_m    = email_results['results'][email_best]

    return {
        "sms_model": {
            "model":     sms_best,
            "accuracy":  sms_m['accuracy'],
            "precision": sms_m['precision'],
            "recall":    sms_m['recall'],
            "f1":        sms_m['f1'],
            "roc_auc":   sms_m['roc_auc']
        },
        "email_model": {
            "model":     email_best,
            "accuracy":  email_m['accuracy'],
            "precision": email_m['precision'],
            "recall":    email_m['recall'],
            "f1":        email_m['f1'],
            "roc_auc":   email_m['roc_auc']
        },
        "behavior_model": {
            "model":              "Isolation Forest",
            "contamination":      0.08,
            "bots_detected":      75,
            "false_positive_rate": "1.1%"
        }
    }