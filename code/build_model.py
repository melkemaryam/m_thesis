# import packages
import argparse
from arguments import Args

# pandas and numpy
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

#import libraries and modules
import io

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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline

class Build_model():

	def __init__(self, model):

		self.model = model

	def get_model(self, C):

		arg = Args()
		args = arg.parse_arguments()

		# train with logistic regression
		if (args["model"] == 'log'):

			# create logistic regression model 
			self.model = LogisticRegression(n_jobs=1, C = C, max_iter = 1000, class_weight="balanced")
			m_name = 'log'

			return self.model, m_name

		# train with support vector machine
		elif (args["model"] == 'svm'):
			
			# create support vector classifier
			self.model = SVC(kernel='linear', probability=True, C = C, gamma = "auto")

			m_name = 'svm'

			return self.model, m_name

		# train with naive bayes
		elif (args["model"] == 'nb'):
			
			# create gaussian naive bayes
			self.model = GaussianNB()

			m_name = 'nb'

			return self.model, m_name

	def get_vector(self, X_train, X_test):

		arg = Args()
		args = arg.parse_arguments()

		# get tfidf
		if (args["vector"] == 'tfidf'):
			
			vector = TfidfVectorizer(max_features=1000, ngram_range=(1, 1))
			v_train = vector.fit_transform(X_train.values.tolist())
			v_test = vector.transform(X_test)
			v_name = 'tfidf'

			return vector, v_train, v_test, v_name

		# get count vectoriser
		elif (args["vector"] == 'count'):
			
			vector = CountVectorizer(max_features=1000, ngram_range=(1, 1))
			v_train = vector.fit_transform(X_train.values.tolist())
			v_test = vector.transform(X_test)
			v_name = 'count'

			return vector, v_train, v_test, v_name

		# train without optimisation
		elif (args["vector"] == 'w2v'):
			
			# Train the word2vec model
			w2v_model = Word2Vec(X_train,
								vector_size=100, ## Size of the Vector
								window=5, ## Number words before and after the focus word that it’ll consider as context for the word
								min_count=2) ## The number of times a word must appear in our corpus in order to create a word vector

			# Transform the data using w2v
			words = set(w2v_model.wv.index_to_key)

			X_train_vect = [np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_train]
			X_test_vect = [np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_test]

			# Compute sentence vectors by averaging the word vectors for the words contained in the sentence
			v_train = []
			for v in X_train_vect:
				if v.size:
					v_train.append(v.mean(axis=0))
				else:
					v_train.append(np.zeros(100, dtype=float))
		            
			v_test = []
			for v in X_test_vect:
				if v.size:
					v_test.append(v.mean(axis=0))
				else:
					v_test.append(np.zeros(100, dtype=float))

			v_name = 'w2v'

			return words, v_train, v_test, v_name


	def build_net(self, data):

		C = 1

		# Features and Labels
		X = data['tokenised']
		y = data['agg_label']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

		# create a classifier 
		self.model, m_name = self.get_model(C)

		vector, v_train, v_test, v_name = self.get_vector(X_train, X_test)

		if v_name == 'w2v':

			self.model.fit(v_train, y_train)
			train_score = self.model.score(v_train, y_train)

			pred = self.model.predict(v_test)
			prob = self.model.predict_proba(v_test)

		else:

			self.model.fit(v_train.toarray(), y_train)
			train_score = self.model.score(v_train.toarray(), y_train)

			pred = self.model.predict(v_test.toarray())
			prob = self.model.predict_proba(v_test.toarray())

		test_score = accuracy_score(y_test, pred)

		print(f"\nShowing results for {v_name} and {m_name} Model, C = {C}")
		print(f"Training Accuarcy: %.3f" % train_score)
		print('Test Accuracy %.3f' % test_score)

		return round(train_score, 3), round(test_score, 3), self.model, vector


	def boost_model(self, data, model):

		arg = Args()
		args = arg.parse_arguments()

		C = 1

		m_name = args["model"]

		# Features and Labels
		X = data['tokenised']
		y = data['agg_label']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

		vector, v_train, v_test, v_name = self.get_vector(X_train, X_test)

		boosting = AdaBoostClassifier(estimator = self.model, 
										n_estimators = 1, 
										learning_rate = 1, 
										random_state = 42)   

		if v_name == 'w2v':

			boosting.fit(v_train, y_train)
			train_score = boosting.score(v_train, y_train)

			pred = boosting.predict(v_test)
			prob = boosting.predict_proba(v_test)

		else:

			boosting.fit(v_train.toarray(), y_train)
			train_score = boosting.score(v_train.toarray(), y_train)

			pred = boosting.predict(v_test.toarray())
			prob = boosting.predict_proba(v_test.toarray())

			
	    
		test_score = accuracy_score(y_test, pred)
	    
		print(f"\nShowing results for {v_name} and {m_name} Model, C = {C}")
		print(f"Training Accuarcy: %.3f" % train_score)
		print('Test Accuracy %.3f' % test_score)

		return train_score, test_score, boosting















