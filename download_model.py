import urllib.request
import os

MODEL_URL = "https://raw.githubusercontent.com/Ahmed-Na7rawy/sleep-health-classifier/master/random_forest_model.pkl"
MODEL_PATH = "random_forest_model.pkl"

def download_model():
    print(f"Downloading model from {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Failed to download model: {e}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        download_model()
    else:
        print("Model already exists.")
