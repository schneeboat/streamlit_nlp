# -*- coding: utf-8 -*-
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud

from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import re
import networkx as nx
from remotezip import RemoteZip

@st.cache
def load_remote():
   with RemoteZip('http://nlp.stanford.edu/projects/glove/glove.6B.zip') as zf:
        file = zf.extract('glove.6B.100d.txt')
   return file
    

st.sidebar.title('Text analysis:')

input_text = st.sidebar.text_area(label="English text you want to analyze:", height=300)  
input_num = st.sidebar.slider(label='How many words to keep for wordcloud viz?', max_value=100)


# #summarization
sent_tok = sent_tokenize(input_text)
formatted_sent_tok = [re.sub('[^a-zA-Z]', ' ', x) for x in sent_tok]
lower_form_sent_tok = [x.lower() for x in formatted_sent_tok]
stop_words=stopwords.words("english")

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
clean_sentences = [remove_stopwords(r.split()) for r in lower_form_sent_tok]


word_embeddings = {}
f = open(load_remote(), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

sentence_vectors = []
for i in sent_tok:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)


sim_mat = np.zeros([len(sent_tok), len(sent_tok)])


for i in range(len(sent_tok)):
  for j in range(len(sent_tok)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sent_tok)), reverse=True)



#cloud
@st.cache
def cloud(text, number):
    formatted_t = re.sub('[^a-zA-Z-]', ' ', text)
    tokenized_text = [word_tokenize(word) for word in [formatted_t]]
    stop_words=stopwords.words("english")
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
    most_freq = keys_freq.most_common(number)
    return keys_freq

    

    
#sentiment analysis   
sia = SentimentIntensityAnalyzer()
score = sia.polarity_scores(input_text)





#output
with st.beta_container():
    st.header('Summary:')
    if not input_text:
        st.warning('Please input the text you want to analyze :)')
    if len(ranked_sentences) >0:
        st.write(ranked_sentences[0][1])
    

  
  
  
with st.beta_container():
     st.header('Wordcloud image')
     if not cloud(input_text, input_num):
         st.warning('Please input the text AND select a number :)')
     if input_num == 0:
         if cloud(input_text, input_num):
             st.warning('Please select a number')
     if (cloud(input_text, input_num)) and (input_num != 0):
          output = cloud(input_text, input_num)
          wordcloud = WordCloud(max_font_size=50, max_words=input_num, background_color='white').generate_from_frequencies(output)  
          fig = plt.figure(figsize=(10,5))
          plt.imshow(wordcloud, interpolation='bilinear')
          plt.axis('off')
          st.pyplot(fig)
     
        
with st.beta_container():    
    st.header('Sentiment analysis')
    
    if input_text:
        fig2, ax = plt.subplots(figsize=(12,6))
        ax.bar(*zip(*score.items()), color='rosybrown', bottom=0)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(height,(p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 4.5), textcoords='offset points', color='black')
            plt.xticks(fontsize=17)
        st.pyplot(fig2)
    if not input_text:
        st.warning('Please input the text you want to analyze :)')

        
        
        
        
        
        
        
        
        
        
        
        
        
