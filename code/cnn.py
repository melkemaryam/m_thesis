import tensorflow.keras as tf
import numpy as np
from keras.layers import Lambda, GlobalAveragePooling1D, Dense, Embedding
from keras import backend as K
from keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

from layers import Layers
from read_data import Read_data
import keras.layers
from pandas import DataFrame
from sklearn.metrics import classification_report
from helper import Helper

class GlobalAveragePooling1DMasked(GlobalAveragePooling1D):
    def call(self, x, mask=None):
        if mask != None:
            return K.sum(x, axis=1) / K.sum(mask, axis=1)
        else:
            return super().call(x)

class Cnn():

	def OneHot(self, input_dim=None, input_length=None):
    
		if input_dim is None or input_length is None:
			raise TypeError("input_dim or input_length is not set")

		def _one_hot(x, num_classes):
			return K.one_hot(K.cast(x, 'uint8'), num_classes=num_classes)

		return Lambda(_one_hot, arguments={'num_classes': input_dim}, input_shape=(input_length,))

	def build_cnn(self, data):

		l = Layers()
		re = Read_data()
		h = Helper()

		word2idx = re.prepare_data(data)
		idx2word = re.get_idx2word(word2idx)
		VOCAB_SIZE = len(word2idx)

		print(idx2word[50])

		MAX_LENGTH = 256
		X_train, X_test, y_train, y_test = l.tokenise(data, MAX_LENGTH)
		print("Training entries: {}, labels: {}".format(len(X_train), len(y_train)))

		print(X_train[1])
		print('\nLength: ',len(X_train))

		model = Sequential()

		# add one-hot layer
		model.add(self.OneHot(input_dim=VOCAB_SIZE, input_length=MAX_LENGTH))

		# compute average
		model.add(GlobalAveragePooling1DMasked())

		# add fully-connected layer
		model.add(Dense(16))

		# add output node with sigmoid
		model.add(Dense(1, "sigmoid"))

		# print model summary
		model.summary()

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		history = model.fit(X_train,
                    y_train,
                    epochs=10,
                    batch_size=64,
                    validation_data=(X_test, y_test),
                    verbose=1)

		pred = model.predict(X_test)
		results = model.evaluate(X_test, y_test)
		labels = h.get_labels()

		report = classification_report(y_test, np.argmax(pred, axis=1), target_names=labels)
		print(report)
		h.write_report(report)
		h.plot_acc(history, X_train, y_train, op = 'normal')
		print('test_loss:', results[0], 'test_accuracy:', results[1])











	
