# import packages
import argparse
from arguments import Args
import os

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras.losses import MSE

from helper import Helper
from read_data import Read_data
from training import Training

class Evaluating():

	def evaluate(self):

		# create objects of class
		h = Helper()
		r = Read_data(None, None, None, None)
		t = Training()

		# get values
		X_test = r.get_x_test()
		y_test = r.get_y_test()

		# evaluate the network
		print("[INFO] evaluating network...")
		model = t.train()

		# get labels
		labels = h.get_labels()

		# test model
		predictions = model.predict(X_test)
		report = classification_report(test_Y.argmax(axis=1), predictions.argmax(axis=1), target_names=image_names)

		print(report)
		h.write_report(report)

		_, acc = model.evaluate(X_test, test_Y, verbose=0)
		print('> %.3f' % (acc * 100.0))
		h.write_report('> %.3f' % (acc * 100.0))