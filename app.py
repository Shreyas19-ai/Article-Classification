from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle

app = Flask(__name__)

ps = PorterStemmer()

words = ['said', 'say', 'like', 'cnn', 'us', 'also']

# Category Mapping
categories = {
    0: 'business',
    1: 'entertainment',
    2: 'health',
    3: 'news',
    4: 'politics',
    5: 'sport'
}

# Load model and vectorizer
cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('xgb30k.pkl', 'rb'))

def txt_process(text):
    text = text.lower()  # Convert to lowercase
    token = nltk.word_tokenize(text)  # Tokenize words

    # Remove stopwords
    y = []
    for i in token:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Remove special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    text = y[:]
    y.clear()

    # Remove custom stop words
    text_str = " ".join(text)
    for i in text_str.split():
        if i not in words:
            y.append(i)

    return " ".join(y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input text from the user
        input_text = request.form['text']

        # 1. Preprocess
        transformed_text = txt_process(input_text)

        # 2. Vectorize
        countvec_input = cv.transform([transformed_text])

        # 3. Predict
        result = model.predict(countvec_input)[0]

        # 4. Map prediction to category
        predicted_category = categories.get(result, "Unknown")

        # 5. Display the result on the page
        return render_template('index.html', prediction_text=f'Predicted Category: {predicted_category}')

if __name__ == "__main__":
    app.run(debug=True)
