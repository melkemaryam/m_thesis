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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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


df_article = pd.read_csv("/Users/Hannah1/Downloads/article_df_final.csv", sep='\t', lineterminator='\n')
df_title = pd.read_csv("/Users/Hannah1/Downloads/title_df_final.csv", sep='\t', lineterminator='\n')

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

## Helper function for training the model and predicting using vectorizer
def get_vector_model(X, y, model = "t", C = 1, v_name = "count", rs = 1):
	# split the dataset
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rs)

	# create a matrix of word counts from the text
	if v_name == "count":
		vector = CountVectorizer(max_features=1000, ngram_range=(1, 1))
	else:
		vector = TfidfVectorizer(max_features=1000, ngram_range=(1, 1))

	# create a classifier 
	if model == "log":

		clf = LogisticRegression(n_jobs=1, C = C, max_iter = 10000, class_weight="balanced")

	elif model == "svm":

		clf = SVC(kernel='linear', probability=True, C = C, gamma = "auto")

	else:
		clf = GaussianNB()

	A = vector.fit_transform(X_train.values.tolist())
	B = vector.transform(X_test)

	# train the classifier with the training data
	if model == "nb":
		clf.fit(A.toarray(), y_train)
		train_score = clf.score(A.toarray(), y_train)
	else:
		clf.fit(A.toarray(), y_train)
		train_score = clf.score(A.toarray(), y_train)


	if model == "nb":
		pred = clf.predict(B.toarray())
		prob = clf.predict_proba(B.toarray())
	else:
		pred = clf.predict(B.toarray())
		prob = clf.predict_proba(B.toarray())

	test_score = accuracy_score(y_test, pred)

	print(f"\nShowing results for {model.title()} Model, C = {C}")

	#show_summary_report(y_test, pred, prob)

	print(f"Training Accuarcy: %.3f" % train_score)
	print('Test Accuracy %.3f' % test_score)

	return round(train_score, 3), round(test_score, 3), clf, vector

#get_vector_model(title_tf['tokenised'], title_tf['agg_label'])


## Helper function for automating building the model using word2vec
def get_w2v_model(df, model = "svm", C = 1, vs = 100, window = 5, mc = 2, rs = 1):
    ## Clean the data using gensim cleaning function
    df['clean_tokenised'] = df['tokenised'].apply(lambda x: gensim.utils.simple_preprocess(x))
    
    # Features and Labels
    X = df['clean_tokenised']
    y = df['agg_label']
    
    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rs)
    
    # create a classifier 
    if model == "log":
        clf = LogisticRegression(n_jobs=1, C = C, max_iter = 1000, class_weight="balanced")
    elif model == "svm":   
        clf = SVC(kernel='linear', probability=True, C = C, gamma = "auto")
    else:
        clf = GaussianNB()
        
    # Train the word2vec model
    w2v_model = Word2Vec(X_train,
                         vector_size=vs, ## Size of the Vector
                         window=window, ## Number words before and after the focus word that it’ll consider as context for the word
                         min_count=mc) ## The number of times a word must appear in our corpus in order to create a word vector
        
    
    ## Transform the data using w2v
    words = set(w2v_model.wv.index_to_key)

    X_train_vect = [np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_train]
    X_test_vect = [np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_test]
    
    # Compute sentence vectors by averaging the word vectors for the words contained in the sentence
    A = []
    for v in X_train_vect:
        if v.size:
            A.append(v.mean(axis=0))
        else:
            A.append(np.zeros(100, dtype=float))
            
    B = []
    for v in X_test_vect:
        if v.size:
            B.append(v.mean(axis=0))
        else:
            B.append(np.zeros(100, dtype=float))
        
    # train the classifier with the training data
    if model == "nb":
        clf.fit(A, y_train)
        train_score = clf.score(A, y_train)
    else:
        clf.fit(A, y_train)
        train_score = clf.score(A, y_train)
    
    
    if model == "nb":
        pred = clf.predict(B)
        prob = clf.predict_proba(B)
    else:
        pred = clf.predict(B)
        prob = clf.predict_proba(B)
    
    test_score = accuracy_score(y_test, pred)
    
    print(f"\nShowing results for {model.title()} Model, C = {C}")
    
    #show_summary_report(y_test, pred, prob)
    
    print(f"Training Accuarcy: %.3f" % train_score)
    print('Test Accuracy %.3f' % test_score)
    
    return round(train_score, 3), round(test_score, 3), clf  


a, b, clf = get_w2v_model(title_tf)

def boost_model(X, y, clf, model = "svm", v_name = "count", n_est = 1, rate = 1, rs = 1):    
    
    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rs)

    # create a matrix of word counts from the text
    if v_name == "count":
        vector = CountVectorizer(max_features=1000, ngram_range=(1, 1))
    else:
        vector = TfidfVectorizer(max_features=1000, ngram_range=(1, 1))
    
    # do the actual counting
    # do the transformation for the test data
    # NOTE: use `transform()` instead of `fit_transform()`
    A = vector.fit_transform(X_train.values.tolist())
    B = vector.transform(X_test)
    
    boosting = AdaBoostClassifier(estimator = clf, 
                                  n_estimators = n_est, 
                                  learning_rate = rate, 
                                  random_state = rs)   
    
    boosting.fit(A, y_train)
    train_score = boosting.score(A, y_train)

    pred = boosting.predict(B)
    prob = boosting.predict_proba(B)
    
    test_score = accuracy_score(y_test, pred)
    
    print(f"\nShowing Boosted Results for {model.title()} Model")
    print(f"Training Accuarcy: %.3f" % train_score)
    print('Test Accuracy %.3f' % test_score)
    
    
    return train_score, test_score, boosting

boost_model(title_tf['tokenised'], title_tf['agg_label'], clf)


'''

def text_representation(data):
  tfidf_vect = TfidfVectorizer()
  #data['tokenised'] = data['tokenised'].apply(lambda text: " ".join(set(text)))
  x_tfidf = tfidf_vect.fit_transform(data['tokenised'])
  print(x_tfidf.shape)
  print(tfidf_vect.get_feature_names_out())
  x_tfidf = pd.DataFrame(x_tfidf.toarray())
  return x_tfidf

#apply the TFIDV function
t_tfidf = text_representation(title_tf)

print(t_tfidf.head())

X= t_tfidf
y = title_tf['agg_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#fit Log Regression Model
clf= LogisticRegression()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

new_data = ["The US imposes sanctions on Russia because of the Ukranian war"]
t = TfidfVectorizer()
tfdf = t.fit_transform(title_tf['tokenised'])
vect = pd.DataFrame(t.transform(new_data).toarray())
new_data = pd.DataFrame(vect)
logistic_prediction = clf.predict(new_data)
print(logistic_prediction)

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

num_epochs = 20
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


'''



