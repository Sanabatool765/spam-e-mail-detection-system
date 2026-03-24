import streamlit as st
import pickle
import string
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# simple stemmer (optional skip bhi kar sakti ho)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = text.split()

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    sw = set(ENGLISH_STOP_WORDS)   # ✅ NLTK hata diya

    for i in text:
        if i not in sw and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# load model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# UI
st.title("📧 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("🚫 Spam")
    else:
        st.header("✅ Not Spam")