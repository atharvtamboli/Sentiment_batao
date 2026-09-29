import pandas as pd
import numpy as np
import re
import html
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# Download NLTK data (runs once)
nltk.download('stopwords')

print("1. Downloading IMDB dataset...")
# Fetching a public IMDB dataset CSV
url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
df = pd.read_csv(url)

# Match the frontend stats: 5000 reviews
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

# Map string sentiments to binary (1 = positive, 0 = negative)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("2. Preprocessing text (this takes a moment)...")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# This EXACTLY matches the clean_review function in your app.py
def clean_review(review: str) -> str:
    review = html.unescape(review)
    review = re.sub(r"<.*?", "", review)
    review = review.lower()
    review = re.sub(r"[^a-z\s]", "", review)
    words = review.split()
    cleaned = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

df['cleaned_review'] = df['review'].apply(clean_review)

print("3. Tokenizing and padding...")
# Convert words to integer sequences
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_review'])

# Save the tokenizer for the Flask backend
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("  ✓ Saved tokenizer.pkl")

# Pad sequences to a uniform length of 100 words
MAXLEN = 100
sequences = tokenizer.texts_to_sequences(df['cleaned_review'])
X = pad_sequences(sequences, padding="post", maxlen=MAXLEN)
y = df['sentiment'].values

print("4. Building the CNN model...")
vocab_size = len(tokenizer.word_index) + 1 

model = Sequential([
    # 100-dimensional embeddings (matches your UI stats)
    Embedding(input_dim=vocab_size, output_dim=100, input_length=MAXLEN), 
    
    # CNN Layer: 128 filters sliding over 5 words at a time (matches your UI stats)
    Conv1D(filters=128, kernel_size=5, activation='relu'),                
    
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5), # Prevents overfitting
    
    # Sigmoid output for a 0 to 1 probability (matches your UI stats)
    Dense(1, activation='sigmoid')                                        
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("5. Training model...")
# Train for 5 Epochs (matches your UI stats)
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

print("6. Saving model...")
model.save("sentiment_batao.h5")
print("  ✓ Saved sentiment_batao.h5")

print("\n🚀 All done! You can now run `python app.py`.")