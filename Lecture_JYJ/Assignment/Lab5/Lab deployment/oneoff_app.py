#!/usr/bin/env python
# coding: utf-8

# In[145]:


import numpy as np  # for manipulation
import pandas as pd  # for data loading

import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

import pickle  # for importing model

from flask import Flask, request, jsonify, render_template  # for handling web service

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

def clean_text(text):
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(' ')), 3) * 100

def preprocessing(data):
	data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(' '))
	data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))
	X_tfidf = fitted_tfidf.transform(data['body_text'])
	X_tfidf_feat = pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)
	return X_tfidf_feat


# model and fitted object loading
model = pickle.load(open('./Lecture_JYJ/Assignment/Lab5/neigh_trained.pkl', 'rb'))
fitted_tfidf = pickle.load(open('./Lecture_JYJ/Assignment/Lab5/tfidf.pkl', 'rb'))
# Flask instantiation
app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return(render_template('index.html'))
    if request.method == 'POST':
        # get input values
        text_input = (request.form['text_message'])
        text_list = []
        text_list.append(text_input)
        text_df = pd.DataFrame(text_list, columns=['body_text'])
        X_tfidf_feat = preprocessing(text_df)
        
        # predict the price
        prediction = model.predict(X_tfidf_feat)

        return render_template('index.html', result=prediction[0])    

# running the application for serving

if __name__ == '__main__':
    app.run(host='127.0.0.1')
