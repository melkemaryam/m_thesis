# create venv python3 -m venv venvo
# start venv source venvo/bin/activate
# 

# pandas and numpy
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

#import libraries and modules
import io

#import warnings
#warnings.filterwarnings('ignore')

#Snorkel
from snorkel.labeling import LabelingFunction
import re
from snorkel.preprocess import preprocessor
from textblob import TextBlob
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling import labeling_function

# API
import requests
import json

# sentiment analysis
import spacy
import nltk
from spacy.lang.en.stop_words import STOP_WORDS
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# NLTK
import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import string
from string import punctuation
from nltk.corpus import stopwords

import nltk.tokenize
punc = string.punctuation
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer  # lemmatization
nltk.download('wordnet')

from statistics import mean
from heapq import nlargest

# miscellaneous
import time as timer
from datetime import datetime, date, time
from tqdm import tqdm
from collections import Counter
import pickle
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt

#Supervised learning
from tqdm import tqdm_notebook as tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

##Deep learning libraries and APIs
import numpy as np
import tensorflow as tf
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences

## Machine Learning Algorithms
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from gensim.models import Word2Vec

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, recall_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import label_binarize
#from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
#from tokenizers import Tokenizer, models
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


df_article = pd.read_csv("/Users/Hannah1/Downloads/article_df_final.csv", sep='\t', lineterminator='\n')
df_title = pd.read_csv("/Users/Hannah1/My Drive/Enigma/m_thesis/info/title_df_final.csv", sep='\t', lineterminator='\n')

df_article.fillna('', inplace=True)
df_article.drop(columns = ['Unnamed: 0'], inplace=True)
df_article = df_article.iloc[:10000]
df_title.fillna('', inplace=True)
df_title.drop(columns = ['Unnamed: 0'], inplace=True)
df_title = df_title.iloc[:10000]


print(df_article.head())
print(df_article.shape)

print(df_title.head())
print(df_title.shape)

# TF IDF


title_tf = df_title.copy()
article_tf = df_article.copy()

max_features = 1000

##store headlines and labels in respective lists
text = list(title_tf['tokenised'])
labels = list(title_tf['agg_label'])

##sentences
training_text = text[0:8000]
testing_text = text[8000:]

##labels
training_labels = labels[0:8000]
testing_labels = labels[8000:]

#preprocess 
tokenizer = Tokenizer(num_words=1000, oov_token= "<OOV>")
tokenizer.fit_on_texts(training_text)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_text)
training_padded = pad_sequences(training_sequences, maxlen=120, padding='post', truncating='post')
testing_sequences = tokenizer.texts_to_sequences(testing_text)
testing_padded = pad_sequences(testing_sequences, maxlen=120, padding='post', truncating='post')

# convert lists into numpy arrays to make it work with TensorFlow 
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=120),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
##compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
 
model.summary()

num_epochs = 30
history = model.fit(training_padded, 
                    training_labels, 
                    epochs=num_epochs, 
                    validation_data=(testing_padded, testing_labels), 
                    verbose=2)

new_headline = ["The US imposes sanctions on Russia because of the Ukranian war"]
##prepare the sequences of the sentences in question
sequences = tokenizer.texts_to_sequences(new_headline)
padded_seqs = pad_sequences(sequences, maxlen=120, padding='post', truncating='post')
print(model.predict(padded_seqs))




