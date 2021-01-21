# -*- coding: utf-8 -*-
import streamlit as st
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import string

st.sidebar.title('Text analysis:')
input_text = st.sidebar.text_area(label="English text you want to analyze:", height=300)
     
input_num = st.sidebar.slider(label='How many words to keep for wordcloud viz?', max_value=500)

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
    fig = plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return st.puplot(fig)

    
    
sia = SentimentIntensityAnalyzer()
score = sia.polarity_scores(input_text)

with st.beta_container():
     st.header('Wordcloud image')
     if not cloud(input_text,input_num):
         st.warning('Please input the text and select a number :)')
        
        
     if cloud(input_text,input_num):
          st.write(cloud(input_text,input_num))
     elif not input_num:
              st.warning('Please select a number :)')
     elif not input_text:
              st.warning('Please input the text you want to analyze :)')
      
      
      
with st.beta_container():    
    st.header('Sentiment analysis')
    
    if input_text:
        fig2, ax = plt.subplots(figsize=(12,6))
        ax.bar(*zip(*score.items()), color='rosybrown', bottom=0)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(height,(p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 4.5), textcoords='offset points', color='black')
        st.pyplot(fig2)
    if not input_text:
        st.warning('Please input the text you want to analyze :)')

        
        
        
        
        
        
        
        
        
        
        
        
        