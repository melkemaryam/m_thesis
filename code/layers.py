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

from read_data import Read_data
from helper import Helper

import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay, learning_curve
from plot_keras_history import plot_history

class Layers():

	def tokenise(self, data):

		re = Read_data()

		X_train, X_test, y_train, y_test = re.train_test_data(data)

		#preprocess 
		tokenizer = Tokenizer(num_words=1000, oov_token= "<OOV>")
		tokenizer.fit_on_texts(X_train)
		word_index = tokenizer.word_index
		X_train = tokenizer.texts_to_sequences(X_train)
		X_train = pad_sequences(X_train, maxlen=120, padding='post', truncating='post')
		X_test = tokenizer.texts_to_sequences(X_test)
		X_test = pad_sequences(X_test, maxlen=120, padding='post', truncating='post')

		# convert lists into numpy arrays to make it work with TensorFlow 
		X_train = np.array(X_train)
		y_train = np.array(y_train)
		X_test = np.array(X_test)
		y_test = np.array(y_test)

		return X_train, X_test, y_train, y_test

	def build_tf(self, data):

		h = Helper()

		# train the network
		print("[INFO] training network...")
		h.write_report(f"The size of this dataset is %.1f" % len(data))

		X_train, X_test, y_train, y_test = self.tokenise(data)

		model = tf.keras.Sequential([
		    tf.keras.layers.Embedding(1000, 64, input_length=120),
		    tf.keras.layers.GlobalAveragePooling1D(),
		    tf.keras.layers.Dense(24, activation='relu'),
		    tf.keras.layers.Dense(1, activation='sigmoid')
		])

		##compile the model
		model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
		 
		model.summary()

		num_epochs = 50
		history = model.fit(X_train, 
		                    y_train, 
		                    epochs=num_epochs, 
		                    validation_data=(X_test, y_test), 
		                    verbose=2)

		pred = model.predict(X_test)
		test_score = model.evaluate(X_test, y_test)
		labels = h.get_labels()

		report = classification_report(y_test, np.argmax(pred, axis=1), target_names=labels)
		print(report)
		h.write_report(report)
		h.plot_acc(history, X_train, y_train, op = 'normal')

		return model

