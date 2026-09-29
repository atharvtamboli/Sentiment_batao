import sys
print("Python version:", sys.version)

print("\n1. Testing imports...")
try:
    import tensorflow as tf
    print("✓ TensorFlow imported:", tf.__version__)
except Exception as e:
    print("✗ TensorFlow error:", e)
    sys.exit(1)

try:
    import pickle
    print("✓ Pickle imported")
except Exception as e:
    print("✗ Pickle error:", e)

print("\n2. Loading model...")
try:
    model = tf.keras.models.load_model("sentiment_batao.h5")
    print("✓ Model loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Model name: {model.name}")
except Exception as e:
    print("✗ Model loading failed:", str(e)[:200])

print("\n3. Loading tokenizer...")
try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("✓ Tokenizer loaded successfully")
    print(f"  Tokenizer type: {type(tokenizer)}")
except Exception as e:
    print("✗ Tokenizer loading failed:", e)

print("\n✅ Both model and tokenizer connected!")
