import spacy
from nltk.stem import PorterStemmer
from rake_nltk import Rake
import nltk
import collections
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from wordcloud import WordCloud, STOPWORDS
import streamlit as st
#import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
#from keybert import KeyBERT

st.set_page_config(layout= "wide")

#Setting the sidebar

st.sidebar.subheader(" :green[NLP]")
st.sidebar.image("nlpgreen.jpg", use_column_width=True)


#Main Page
st.title(" :green[Natural Language Processing]")
text = st.text_input("Enter the text:")

col1,col2 = st.columns(2)
col3,col4 = st.columns(2)
col5,col6 = st.columns(2)

with col1:
#text = "NLP is a subfield of computer science and artificial intelligence that deals with the interaction between computers and human language."
    Stemming = st.button("Stemming")
    if Stemming:

        # Create a stemmer object
        stemmer = PorterStemmer()

        # Stem each word in the text
        stems = [stemmer.stem(word) for word in text.split()]

        # Print the stems
        st.write("After Stemming the Input Text")
        st.write(stems)
         
with col2:
    NER = st.button("NER") 
    if NER:
        # Tokenize and tag the text
        sentences = nltk.sent_tokenize(text)
        tagged_sentences = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in sentences]

        # Perform NER with binary classification
        binary_ner = nltk.chunk.ne_chunk_sents(tagged_sentences, binary=True)

        # Print the binary NER results
        for sent in binary_ner:
            st.write("Name Entity Recogination")
            st.write(sent)

with col3:
    keyword= st.button("Keyword")
    if keyword:
        r = Rake()
    #my_text = "NLP is a subfield of computer science and artificial intelligence that deals with the interaction between computers and human language."
        r.extract_keywords_from_text(text)
        keywordList           = []
        rankedList            = r.get_ranked_phrases_with_scores()
        for keyword in rankedList:
            keyword_updated       = keyword[1].split()
            keyword_updated_string    = " ".join(keyword_updated[:2])
            keywordList.append(keyword_updated_string)
            if(len(keywordList)>9):
                break
        st.write(keywordList)      

with col4:
    wordcloud = st.button("Word Cloud")  
    if wordcloud:
        stopwords = STOPWORDS
        wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=1000).generate(text)
        rcParams['figure.figsize'] = 10, 20
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
        filtered_words = [word for word in text.split() if word not in stopwords]
        counted_words = collections.Counter(filtered_words)
        words = []
        counts = []
        for letter, count in counted_words.most_common(10):
            words.append(letter)
            counts.append(count)
        colors = cm.rainbow(np.linspace(0, 1, 10))
        rcParams['figure.figsize'] = 20, 10
        plt.title('Word Cloud')
        plt.xlabel('Count')
        plt.ylabel('Words')
        plt.barh(words, counts, color=colors)
        st.pyplot(plt.gcf())        

with col5:
    SentimentAnalysis = st.button("Sentiment Analysis")       
    if SentimentAnalysis:
        # Instantiate the SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()

        # Analyze sentiment for each sentence
        sentiment_scores = vader.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        if compound_score > 0.05:
            sentiment = "Positive"
        elif compound_score < -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        st.write(text)
        st.write(sentiment ,compound_score)
        st.write("-" * 30)
