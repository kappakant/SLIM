import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import csv

# n = 50 => Top 50% of TFIDF keywords
n = 50

training   = pd.read_csv('~/Desktop/4thSem/DATALab/fake_and_real_news/train.csv')
testing    = pd.read_csv('~/Desktop/4thSem/DATALab/fake_and_real_news/test.csv')
validation = pd.read_csv('~/Desktop/4thSem/DATALab/fake_and_real_news/val.csv')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

### Cleaning datasets
import numpy as np
training.replace('', np.nan, inplace=True)
training.replace(' ', np.nan, inplace=True)
training.dropna(inplace=True)
training = training.reset_index(drop=True)

testing.replace('', np.nan, inplace=True)
testing.replace(' ', np.nan, inplace=True)
testing.dropna(inplace=True)
testing = testing.reset_index(drop=True)

validation.replace('', np.nan, inplace=True)
validation.replace(' ', np.nan, inplace=True)
validation.dropna(inplace=True)
validation = validation.reset_index(drop=True)

################# Check the Datasets ################# 
print(training.head(2))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
    return ' '.join(tokens)

training['body_text']   = training['body_text'].apply(preprocess_text)
testing['body_text']    = testing['body_text'].apply(preprocess_text)
validation['body_text'] = validation['body_text'].apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer()
def TFIDF_into_selected_words(doc):
    selected_words = []
    for i in range(len(doc)):
        doc_tfidf_matrix = tfidf_vectorizer.fit_transform(doc['body_text'])
        vector = doc_tfidf_matrix[i]
        dense_vector = vector.T.todense()
        dense_vector = dense_vector.A1
        word_count_10_percent = int(len(doc['body_text'][i].split()) * (n / 100))
        top_indices = dense_vector.argsort()[-word_count_10_percent:][::-1]
        feature_names = tfidf_vectorizer.get_feature_names_out()
        selected_words.append(' '.join(feature_names[top_indices]))
    doc['selected_words'] = selected_words

TFIDF_into_selected_words(training)
TFIDF_into_selected_words(testing)
TFIDF_into_selected_words(validation)

training.to_csv(f'~/Desktop/4thSem/DATALab/fake_and_real_news/TFIDFtrain{n}.csv',index=False)
testing.to_csv(f'~/Desktop/4thSem/DATALab/fake_and_real_news/TFIDFtest{n}.csv',index=False)
validation.to_csv(f'~/Desktop/4thSem/DATALab/fake_and_real_news/TFIDFval{n}.csv',index=False)
