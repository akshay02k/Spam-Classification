from flask import Flask, request, jsonify
import requests
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer




with open("rmc_model.pkl","rb") as model_file:
    rmc_model = pickle.load(model_file)


with open("tfidf_vectorizer.pkl","rb") as model_file:
    tfidf_vectorizer = pickle.load(model_file)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]*>", "", text)
    # tokenization
    token = word_tokenize(text)
    # remove punctuation
    clean_token = [word for word in token if word.isalnum()]

    # remove stopwords
    stopwords_list = set(stopwords.words("english"))
    filter_tokens = [word for word in clean_token if word not in stopwords_list]

    # stemming
    ps = PorterStemmer()
    stem_tokens = [ps.stem(word) for word in filter_tokens]

    clean_text=  ' '.join(stem_tokens)
    return clean_text



app = Flask(__name__)
@app.route("/predict", methods = ["POST"])
def predict():
    message = request.json["message"]

    preprocessed_text = preprocess_text(message)
    vectorized_mesage= tfidf_vectorizer.transform([preprocessed_text])

    prediction = rmc_model.predict(vectorized_mesage)
    return jsonify({"prediction": prediction[0]})


if __name__ == "__main__":
    app.run(host = "0.0.0.0",port = 5000, debug=True)