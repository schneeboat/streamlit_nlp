# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import string


input_text = st.sidebar.text_area(label="Text you want to analyze:")
input_num = st.sidebar.slider(label='How many words to keep (max)?', max_value=500)

def cloud(text, number):
    
    tokenized_text = [word_tokenize(word) for word in [text]]
    stop_words=set(stopwords.words("english")+list(string.punctuation)+list(string.digits))
    filtered_texts=[]
    for t in tokenized_text:
        filtered_text=[]
        for word in t:
            if word.lower() not in stop_words:
                filtered_text.append(word)
        filtered_texts.append(filtered_text)

    keywords_text = [[item.lower() for item in sublist] for sublist in filtered_texts]
    kl = [x for sub in keywords_text for x in sub]
    keys_freq = Counter(kl)
    #most_freq_key = keys_freq.most_common(number)
    
    wordcloud = WordCloud(max_font_size=50, max_words=number, background_color='white').generate_from_frequencies(keys_freq)
    fig = plt.subplots()
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return fig
if cloud(input_text,input_num):
    st.write(cloud(input_text,input_num))





