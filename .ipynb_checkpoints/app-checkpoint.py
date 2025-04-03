from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('model.pkl')
countvector = joblib.load('countvector.pkl')
tfidfvector = joblib.load('tfidfvector.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['text']
    combined_text = data  # Assuming the input is already combined title and abstract

    # Preprocess the input text
    combined_text = combined_text.lower()
    combined_text = ' '.join([word for word in combined_text.split() if len(word) > 1])

    # Vectorize the input text
    X_cv = countvector.transform([combined_text])
    X_tf = tfidfvector.transform(X_cv)

    # Make prediction
    prediction = model.predict(X_tf)
    prediction = prediction[0]  # Since it's a single prediction

    # Map prediction to labels
    labels = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
    result = {label: int(pred) for label, pred in zip(labels, prediction)}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)