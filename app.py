from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from your frontend

# ── Load model & tokenizer once at startup ──────────────────────────────────
print("Loading sentiment model and tokenizer...")

model = None
tokenizer = None

try:
    model = load_model("sentiment_batao.h5")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")

try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"✗ Error loading tokenizer: {e}")

if model is None:
    print("❌ Failed to load model - backend will not work")
    MAXLEN = 100  # Default fallback
else:
    MAXLEN = model.input_shape[1]

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))



# ── Preprocessing (mirrors your training code) ───────────────────────────────
def clean_review(review: str) -> str:
    review = html.unescape(review)
    review = re.sub(r"<.*?", "", review)
    review = review.lower()
    review = re.sub(r"[^a-z\s]", "", review)
    words = review.split()
    cleaned = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(cleaned)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    review = data.get("review", "").strip()

    if not review:
        return jsonify({"error": "No review text provided."}), 400

    cleaned = clean_review(review)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, padding="post", maxlen=MAXLEN)

    raw_score = float(model.predict(padded)[0][0])
    sentiment = "positive" if raw_score > 0.5 else "negative"
    confidence = raw_score if raw_score > 0.5 else 1 - raw_score

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "raw_score": round(raw_score, 4),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "sentiment_batao.h5"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
