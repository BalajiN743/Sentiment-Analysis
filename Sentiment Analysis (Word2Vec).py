# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:56:53 2023

@author: 91944
"""

import pandas as pd
import numpy as np
import streamlit as st 
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lm=WordNetLemmatizer()
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
le=LabelEncoder()
import pickle
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
import gensim.downloader 
import gensim
from gensim.models import Word2Vec, KeyedVectors

#Text Cleaning
def text_cleaning(Text):
    
    #removal of the link
    Text = re.sub('https?://\S+|www\.\S+', '', Text)
    #removal of punctuatoins
    punc=str.maketrans(string.punctuation,' '*len(string.punctuation))
    Text=Text.translate(punc)
    #removal of numbers
    Text = re.sub(r'\d+', '', Text)
    #removal of the special characters
    Text=re.sub(r'[^\w\s]', '', Text) 
    #lower case transformation
    Text=Text.lower()
    #remove the non-english alphabets
    Text=re.sub(r'[^\u0000-\u007F]+', '', Text)
    
    return Text

Stopwords = (set(nltk.corpus.stopwords.words("english")))
Stopwords.remove('not')
Stopwords.remove('down')
Stopwords.remove("more")
Stopwords.remove("under")
domain_words=['finnish','russian','finland','russia','swedish','firm','eighteen','months','taking','total','square',
              'eur','million','announcement','day','earlier','glaston','net','third','quarter','dropped','mln','euro',
              'period','april','baltic','countries','eur mn','last','year','million','state',
              'office','msft','orcl','goog','crm','adbe','aapl','afternoon','esi','billion','eurm','third','quarter',
              'half','annually','annualy','first','second','nine','helsinki','omx','year','month','day','indian','india','third'
              ,'fourth','mn','mln','in','eur','euro','months','goods','one','the', 'of', 'in', 'to', 'and', 'a','eur', 'for',
              's', 'is', 'on', 'from', 'will', 'company', 'as', 'mn', 'its', 'with', 'by', 'be', 'has', 'at','it', 'said', 
              'million', 'net', 'year', 'm', 'that', 'was', 'group', 'an', 'mln','new', 'are', 'quarter','this', 'oyj','also',
              'have', 'which', 'first', 'euro', 'today', 'been', 'about', 'helsinki', 'per','total', 'after', 'nokia', 'bank', 
              'based', 'were', 'we', 'than', 'some','or', 'other', 'all', 'one', 'hel' ,'our', 'plc', 'now', 'last', 'their',
              'second', 'ceo', 'pct', 'january', 'into', 'aapl', 'would', 'eurm', 'out', 'part', 'oy','i','september', 'usd',
              'two', 'third','earlier', 'can', 'time', 'billion','had', 'omx','us', 'russia', 'may','annual', 'day', 'both', 
              'tsla','while', 'before','months', 'number', 'march', 'october', 'euros',
              'they','through', 'april']
Stopwords.update(domain_words)

def Text_Processing(Text):    
    Processed_Text = list()
    Lemmatizer = WordNetLemmatizer()

    # Tokens of Words
    Tokens = nltk.word_tokenize(Text)

    for word in Tokens:
        if word not in Stopwords:            
            Processed_Text.append(Lemmatizer.lemmatize(word))            
    return(" ".join(Processed_Text))

model = pickle.load(open('W2V_model.pkl','rb'))
RF_clf = pickle.load(open('RF_W2V_clf.pkl', 'rb'))

st.title("Sentiment Analyzer")
st.subheader('for financial Texts')
text_input = st.text_area("Enter the message")

if st.button('Analyze'):
    cleaned_text = text_cleaning(text_input)
    transform_review=Text_Processing(cleaned_text)
    sent_vec = np.zeros(100)
    count = 0
    for word in transform_review.split():
        if word in model.wv.key_to_index:
            vec = model.wv[word]
            sent_vec += vec
            count += 1
    if count != 0:
        sent_vec /= count
    text_vec = sent_vec.reshape(1, -1)  # reshape to match the expected input dimensions
    # predict
    result = RF_clf.predict(text_vec)
    # Display
    if result == 0:
        st.header('Negative Statement')
    elif result == 1:
        st.header('Neutral statement')
    elif result == 2:
        st.header('Positive statement')


