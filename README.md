## SENTIMENT BATAO

A high-performance sentiment analysis web application featuring a **Convolutional Neural Network (CNN)** backend. This tool dissects movie reviews (trained on the IMDB dataset) to expose the emotion hiding within the text.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

## Features
* **Live Neural Inference:** Real-time sentiment prediction using a trained `.h5` model.
* **Futuristic UI:** Custom cursor, particle background, and GSAP animations for a premium UX.
* **Deep NLP Pipeline:** Implements HTML unescaping, stopword removal, and Porter Stemming.
* **Confidence Metrics:** Visualizes the sigmoid output with dynamic confidence bars.
* **History Tracking:** Persists your recent analyses using LocalStorage.

## Technical Architecture
### Backend (Flask + TensorFlow)
* **Model:** 1D Convolutional Neural Network (Conv1D) with 128 filters.
* **Embeddings:** 100-dimensional word embeddings.
* **Preprocessing:** NLTK-based cleaning pipeline to reduce noise.
* **Output:** Sigmoid activation providing a probability score between 0 and 1.

### Frontend (HTML5 + CSS3 + JS)
* **Animations:** GSAP (GreenSock) and ScrollTrigger.
* **Background:** Particles.js for a dynamic "neural network" feel.
* **Typography:** 'Bebas Neue' and 'Space Mono' for a technical, glitch-hop aesthetic.

## Getting Started

### Prerequisites
* Python 3.8+
* `sentiment_batao.h5` (The trained model file)
* `tokenizer.pkl` (The pickled tokenizer)

### Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/sentiment-batao.git](https://github.com/your-username/sentiment-batao.git)
   cd sentiment-batao
   
Install Dependencies:
pip install -r requirements.txt
Run the Backend:
python app.py







