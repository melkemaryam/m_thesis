# import packages
import argparse
from arguments import Args

# pandas and numpy
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

#import libraries and modules
import io

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold 
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import gensim
from gensim.models import Word2Vec
from tqdm import tqdm_notebook as tqdm

from read_data import Read_data
from helper import Helper


class Build_sk():

	def __init__(self, model):

		self.model = model

	def return_model(self):

		arg = Args()
		args = arg.parse_arguments()
		C = 1

		if (args["model"] == 'tf'):

			# train with all optimisations
			if (args["train"] == 'all'):
				return self.build_net_opt

			# train with one optimisation
			elif (args["train"] == 'one'):
				return self.build_net

			# train without optimisation
			elif (args["train"] == 'none'):
				return self.build_net

		else:

			# train with logistic regression
			if (args["model"] == 'log'):

				# create logistic regression model 
				model = LogisticRegression(n_jobs=1, C = C, max_iter = 1000, class_weight="balanced")

				return model

			# train with support vector machine
			elif (args["model"] == 'svm'):
				
				# create support vector classifier
				model = SVC(kernel='linear', probability=True, C = C, gamma = "auto")

				return model

			# train with naive bayes
			elif (args["model"] == 'nb'):
				
				# create gaussian naive bayes
				model = GaussianNB()

				return model

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

				v_train = v_train.toarray()
				v_test = v_test.toarray()

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

			return words, v_train, v_test

			
































