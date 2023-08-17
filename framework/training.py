from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import MSE

from preprocessing import Preprocessing
from helper import Helper
from build_sk import Build_sk
from tuning import Tuning
from build_tf import Build_tf
from read_data import Read_data

from datetime import datetime


# import packages
import argparse
from arguments import Args

import joblib

class Training():

	def train_sk(self):

		r = Read_data()
		h = Helper()
		sk = Build_sk(None)
		arg = Args()
		args = arg.parse_arguments()

		# get values
		X_train, X_test, y_train, y_test = r.train_test_data()
		print(X_train.shape,y_train.shape)
		print(X_test.shape,y_test.shape)

		# train the network
		print("[INFO] training network...")
		h.write_report(f"The size of this dataset is %.1f" % ((len(X_train) + len(X_test) + len(y_train) + len(y_test))/2))

		vector, v_train, v_test = sk.get_vector(X_train, X_test)
		model = sk.return_model()

		model.fit(v_train, y_train)
		train_score = model.score(v_train, y_train)

		pred = model.predict(v_test)
		prob = model.predict_proba(v_test)
		test_score = accuracy_score(y_test, pred)
		labels = h.get_labels()

		report = classification_report(y_test, pred, target_names=labels)
		h.write_score(train_score, test_score)
		print(report)
		h.write_report(report)
		h.plot_acc(model, v_train, y_train, 'normal')
		h.plot_loss(model)
		print("Confidence for each prediction: " + str(prob))
		h.write_report("Confidence for each prediction: " + str(prob))

		# save the network to disk
		print("[INFO] serializing network to '{}'...".format(args["path"]))
		joblib.dump(model, args["path"])

		return model

	def train_tf(self):

		re = Read_data()
		h = Helper()
		b = Build_tf(None)
		t = Tuning(None)
		arg = Args()
		args = arg.parse_arguments()

		MAX_LENGTH = 256

		X_train, X_test, y_train, y_test, tok = re.tokenise(MAX_LENGTH)
		print(X_train.shape,y_train.shape)
		print(X_test.shape,y_test.shape)

		best_hyperparameters = t.get_best_parameters()

		# train the network
		print("[INFO] training network...")
		h.write_report(f"The size of this dataset is %.1f" % ((len(X_train) + len(X_test) + len(y_train) + len(y_test))/2))

		# create logs for Tensorboard
		log_dir = "../logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

		# create Early Stopping
		callback = EarlyStopping(monitor='loss', patience=3)

		# get the correct model
		if (args["train"] == 'all' or args["train"] == 'one'):
			tuner = t.return_tuner()
			model = tuner.hypermodel.build(best_hyperparameters)
		elif (args["train"] == 'none' or args["train"] == 'pred'):
			model = b.build_net()

		model.summary()

		# train the model
		history = model.fit(X_train,
			y_train,
			validation_data=(X_test, y_test),
			epochs=10,
			batch_size=64, 
			callbacks=[callback, tensorboard],
			verbose=1)

		# save the network to disk
		print("[INFO] serializing network to '{}'...".format(args["path"]))
		model.save(args["path"])

		# evaluate the network
		print("[INFO] evaluating network...")

		# get labels
		labels = h.get_labels()

		# test model
		predictions = model.predict(X_test)
		results = model.evaluate(X_test, y_test)
		report = classification_report(y_test.argmax(axis=1), np.argmax(predictions, axis=1), target_names=labels)

		print(report)
		h.write_report(report)

		_, acc = model.evaluate(X_test, y_test, verbose=0)
		print('> %.3f' % (acc * 100.0))
		h.write_report('> %.3f' % (acc * 100.0))
		h.plot_acc(history, X_train, y_train, op = 'normal')
		print('test_loss:', results[0], 'test_accuracy:', results[1])
		h.write_score(acc, results[1])

		print("Confidence for each prediction: " + str(predictions))
		h.write_report("Confidence for each prediction: " + str(predictions))

		return model


