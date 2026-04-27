# Sleep Health Risk Classifier

This Streamlit web application predicts sleep health outcomes using a Machine Learning model (Random Forest). It allows users to input their physiological and lifestyle metrics and provides a prediction regarding their sleep quality or potential sleep disorders.

![Screenshot](https://via.placeholder.com/800x400.png?text=App+Screenshot+Placeholder)

## What it Predicts
The application utilizes a trained machine learning model to predict **Sleep Disorder** risk (e.g., Insomnia, Sleep Apnea, or None) based on features like age, sleep duration, quality of sleep, physical activity level, heart rate, and daily steps.

## Tech Stack
- **Languages:** Python
- **Framework:** Streamlit
- **Machine Learning:** Scikit-learn, Random Forest

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ahmed-Na7rawy/sleep-health-classifier.git
   cd sleep-health-classifier
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the pre-trained model:**
   Since the model file is large, it's not stored in the repository. Run the download script to fetch it:
   ```bash
   python download_model.py
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Dataset
The model is trained on the [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) from Kaggle.

## Repository Structure
- `app.py`: Main Streamlit application file.
- `notebooks/sleep_health_ml_analysis.ipynb`: The Jupyter notebook containing the full data analysis, model training, and evaluation.
- `download_model.py`: Script to download the `.pkl` model file.
- `requirements.txt`: Python dependencies.
