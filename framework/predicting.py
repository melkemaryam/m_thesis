"""

class: Predicting

methods:
* load_net()
* prediction_process()

purpose:
* make predictions using a specific model

"""

# import packages
import argparse
from arguments import Args

import numpy as np
import os
import random

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from tensorflow.keras.models import load_model
from helper import Helper
from read_data import Read_data

import joblib

class Predicting():

	def load_net(self):

		# create objects of classes
		arg = Args()
		args = arg.parse_arguments()

		# load the trained model
		print("[INFO] loading model...")

		if(args["model"] == 'basic' or args["model"] == 'cnn' or args["model"] == 'lstm' or args["model"] == 'bilstm' or args["model"] == 'rnn'):
			model = load_model(args["path"])

		elif(args["model"] == 'log' or args["model"] == 'svm' or args["model"] == 'nb'):
			model = joblib.load(args["path"])

		return model

	def prediction_process(self, **data):

		# create objects of classes
		arg = Args()
		args = arg.parse_arguments()
		h = Helper()
		r = Read_data()

		# grab the paths to the input images, shuffle them, and grab a sample
		print("[INFO] predicting...")
		h.write_report("[INFO] predicting...")

		labels = h.get_labels()
		MAX_LENGTH = 256
		model = self.load_net()

		if(args["model"] == 'basic' or args["model"] == 'cnn' or args["model"] == 'lstm' or args["model"] == 'bilstm' or args["model"] == 'rnn'):
			X_train, X_test, y_train, y_test = r.train_test_data()
			tokeniser = Tokenizer(num_words=10000, oov_token= "<OOV>")
			tokeniser.fit_on_texts(X_train)

		elif(args["model"] == 'log' or args["model"] == 'svm' or args["model"] == 'nb'):
			# get values
			X_train, X_test, y_train, y_test = r.train_test_data()
			tokeniser = Tokenizer(num_words=10000, oov_token= "<OOV>")
			tokeniser.fit_on_texts(X_train)
			MAX_LENGTH = 1000

		for x in X_test[:10]:

			print(x)
			h.write_report(x)
			sequences = tokeniser.texts_to_sequences(x)
			padded_seqs = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
			p = model.predict(padded_seqs)
			
			j = p.argmax(axis=1)[0]
			label = labels[j]

			print(p)
			h.write_report(p)

			print(j)
			h.write_report(j)

			print(label)
			h.write_report(label)
			
			print("Confidence for each prediction: " + str(p[0]))
			h.write_report("Confidence for each prediction: " + str(p[0]))
