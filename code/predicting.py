import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
from helper import Helper
from read_data import Read_data

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

class Predicting():

	def prediction_process(self, data, model, tokeniser):

		h = Helper()
		re = Read_data()

		print("[INFO] predicting...")
		h.write_report("[INFO] predicting...")

		labels = h.get_labels()

		X_train, X_test, y_train, y_test = re.train_test_data(data)

		for x in X_test[:10]:

			print(x)
			h.write_report(x)
			sequences = tokeniser.texts_to_sequences(x)
			padded_seqs = pad_sequences(sequences, maxlen=256, padding='post', truncating='post')
			p = model.predict(padded_seqs)
			print(p[0])
			h.write_report(p[0])

