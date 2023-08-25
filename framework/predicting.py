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
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from tensorflow.keras.models import load_model
from helper import Helper
from read_data import Read_data

import joblib

# Define a class named Predicting
class Predicting():

	# load the correct model for predictions
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

	# predict labels using a model
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
		model = self.load_net()

		# use Australian news for predictions
		new_data = pd.read_csv("../abcnews-date-text.csv")
		new_data = new_data['headline_text']

		tokeniser = Tokenizer(num_words=10000, oov_token= "<OOV>")
		tokeniser.fit_on_texts(new_data)

		if(args["model"] == 'basic' or args["model"] == 'cnn' or args["model"] == 'lstm' or args["model"] == 'bilstm' or args["model"] == 'rnn'):
			
			MAX_LENGTH = 256

		elif(args["model"] == 'log' or args["model"] == 'svm' or args["model"] == 'nb'):
	
			MAX_LENGTH = 1000

		pos_count = 0
		neg_count = 0
		neu_count = 0
		x_count = 0

		# limit to 100 predictions
		for x in new_data[:100]:

			print(x)
			h.write_report(x)
			sequences = tokeniser.texts_to_sequences(x)
			padded_seqs = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
			p = model.predict(padded_seqs)

			if(args["model"] == 'basic' or args["model"] == 'cnn' or args["model"] == 'lstm' or args["model"] == 'bilstm' or args["model"] == 'rnn'):
			
				j = p.argmax(axis=1)[0]
				label = labels[j]

				print(p[0])
				h.write_report(p[0])

				print(p)
				h.write_report(p)

				print(j)
				h.write_report(j)

				print(label)
				h.write_report(label)

				h.write_pred(j, x)

				if j == 0:
					neg_count += 1

				elif j == 1:
					pos_count += 1

				elif j == 2:
					neu_count += 1

				x_count += 1

			elif(args["model"] == 'log' or args["model"] == 'svm' or args["model"] == 'nb'):
	
				print(p)
				h.write_report(p)

		print("Number of negative predictions: " + str(neg_count))
		h.write_report("Number of negative predictions: " + str(neg_count))
		print("Number of positive predictions: " + str(pos_count))
		h.write_report("Number of positive predictions: " + str(pos_count))
		print("Number of neutral predictions: " + str(neu_count))
		h.write_report("Number of neutral predictions: " + str(neu_count))
		print("Number of all predictions: " + str(x_count))
		h.write_report("Number of neutral predictions: " + str(x_count))
		print("Model used: " + str(args["model"]))
		h.write_report("Model used: " + str(args["model"]))
