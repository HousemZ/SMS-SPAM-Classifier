# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 23:45:02 2022

@author: dell
"""
import streamlit as st
import pandas as pd
import pickle
import time
import re
import nltk
from nltk.stem import PorterStemmer
from PIL import Image

stop_words = nltk.corpus.stopwords.words("english")
stemmer = PorterStemmer()
nltk.download("stopwords")

def transform_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    l_text = nltk.tokenize.word_tokenize(text)
    l_word = [word for word in l_text if not word in stop_words]
    stemmed_words = [stemmer.stem(word) for word in l_word]
    return " ".join(stemmed_words)


mdl = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vect.pkl","rb"))

st.title('Msg spam classifier')

image = Image.open('spam.jpg')

st.image(image, caption='Msg spam classifier')


input_text = st.text_area("Please enter your msg ", height=100)
if st.button("Send"):
    transf_msg = transform_text(input_text)
    X = vectorizer.transform([transf_msg])
    progress_bar = st.progress(0)
    progress_text = st.empty()
    for i in range(0,101,20):
        time.sleep(0.05)
        progress_bar.progress(i)
        progress_text.text(f"Progress: {i}%")
    result = mdl.predict(X)
    resu_p = mdl.predict_proba(X)
    if result==1:
        st.metric("Result", "SPAM msg",delta=round(resu_p[:,1][0],ndigits=2))
    else:
        st.metric("Result", "Not SPAM",delta=round(resu_p[:,0][0],ndigits=2))
    prob = pd.DataFrame(resu_p, columns=['Not spam', 'SPAM'])
    st.bar_chart(prob,use_container_width=True, width=600, height=300)
    




