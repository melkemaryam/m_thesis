"""

class: Build_sk

methods:
* return_model()
* get_vector()

purpose:
* build a model using the sci-kit learn library

"""

# import packages
import argparse
from arguments import Args

# pandas and numpy
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

#import libraries and modules
import io

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from gensim.models import Word2Vec

from read_data import Read_data
from helper import Helper


class Build_sk():

	def __init__(self, model):

		self.model = model

	def return_model(self):

		arg = Args()
		args = arg.parse_arguments()

		if (args["model"] == 'log' or args["model"] == 'svm' or args["model"] == 'nb'):

			# train with logistic regression
			if (args["model"] == 'log'):

				# create logistic regression model 
				self.model = LogisticRegression(n_jobs=1, C = 1, max_iter = 1000, class_weight="balanced")

				return self.model

			# train with support vector machine
			elif (args["model"] == 'svm'):
				# create support vector classifier
				self.model = SVC(kernel='linear', probability=True, C = 1, gamma = "auto")

				return self.model

			# train with naive bayes
			elif (args["model"] == 'nb'):
				# create gaussian naive bayes
				self.model = GaussianNB()

				return self.model

	def get_vector(self, X_train, X_test):

		# get arguments
		arg = Args()
		args = arg.parse_arguments()

		# get tfidf
		if (args["vector"] == 'tfidf'):
			
			# create tfidf vector
			vector = TfidfVectorizer(max_features=1000, ngram_range=(1, 1))
			v_train = vector.fit_transform(X_train.values.tolist())
			v_test = vector.transform(X_test)

			# change type to array if model is naive bayes
			if (args["model"] == 'nb'):

				v_train = v_train.reshape(-1, 1)
				v_test = v_test.reshape(-1, 1)

			return vector, v_train, v_test

		# get count vectoriser
		elif (args["vector"] == 'count'):
			
			vector = CountVectorizer(max_features=1000, ngram_range=(1, 1))
			v_train = vector.fit_transform(X_train.values.tolist())
			v_test = vector.transform(X_test)

			# change type to array if model is naive bayes
			if (args["model"] == 'nb'):

				v_train = v_train.toarray()
				v_test = v_test.toarray()
			
			return vector, v_train, v_test

		# train without optimisation
		elif (args["vector"] == 'w2v'):
			
			# Train the word2vec model
			w2v_model = Word2Vec(X_train,
								vector_size=6000, ## Size of the Vector
								window=5, ## Number words before and after the focus word that it’ll consider as context for the word
								min_count=4) ## The number of times a word must appear in our corpus in order to create a word vector

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

			# change type to array if model is naive bayes
			if (args["model"] == 'nb'):

				v_train = v_train.toarray()
				v_test = v_test.toarray()

			return words, v_train, v_test
