# import packages
import argparse
from arguments import Args

import numpy as np
import os
import random

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

		if(args["model"] == 'basic' or args["model"] == 'cnn' or args["model"] == 'lstm' or args["model"] == 'bilstm'):
			model = load_model(args["path"])

		elif(args["model"] == 'log' or args["model"] == 'svm' or args["model"] == 'nb'):
			model = joblib.load(args["path"])

		return model

	def prediction_process(self, **data):

		# create objects of classes
		arg = Args()
		args = arg.parse_arguments()
		h = Helper()
		r = Read_data(None, None, None, None, None)

		# grab the paths to the input images, shuffle them, and grab a sample
		print("[INFO] predicting...")
		h.write_report("[INFO] predicting...")

		labels = h.get_labels()
		MAX_LENGTH = 256

		if(args["model"] == 'basic' or args["model"] == 'cnn' or args["model"] == 'lstm' or args["model"] == 'bilstm'):
			X_train, X_test, y_train, y_test, tok = re.tokenise(MAX_LENGTH)

		elif(args["model"] == 'log' or args["model"] == 'svm' or args["model"] == 'nb'):
			# get values
			X_train = r.get_x_train()
			X_test = r.get_x_test()
			y_train = r.get_y_train()
			y_test = r.get_y_test()

		for x in X_test[:10]:

			print(x)
			h.write_report(x)
			sequences = tokeniser.texts_to_sequences(x)
			padded_seqs = pad_sequences(sequences, maxlen=256, padding='post', truncating='post')
			p = model.predict(padded_seqs)
			print(float(p[0]))
			h.write_report(float(p[0]))
			print("Confidence for each prediction: " + str(p))
			h.write_report("Confidence for each prediction: " + str(p))
