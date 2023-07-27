from read_data import Read_data
from helper import Helper
from layers import Layers

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, LSTM, Dropout
from keras.models import Model
from plot_keras_history import plot_history
from pandas import DataFrame
from sklearn.metrics import classification_report
import numpy as np


class Lstm_clf():

	def build_lstm(self, data):

		re = Read_data()
		l = Layers()
		h = Helper()
		word2idx = re.prepare_data(data)
		MAX_LENGTH = 200

		X_train, X_test, y_train, y_test = l.tokenise(data, MAX_LENGTH)

		print('Length of sample train_data after preprocessing:', len(X_train[0]))
		print('Sample train data:', X_train[0])

		EMBED_SIZE = 100
		VOCAB_SIZE = len(word2idx)

		# create the input layer
		in_layer = Input((MAX_LENGTH,), dtype='int32')

		# add the embedding layer
		emb_layer = Embedding(VOCAB_SIZE, EMBED_SIZE, mask_zero=True, input_length = MAX_LENGTH)(in_layer)

		# add first dropout layer
		drop_1 = Dropout(0.5)(emb_layer)

		# add LSTM
		# return_sequences is already FALSE as default
		lstm_layer = LSTM(100)(drop_1)

		# add first dropout layer
		drop_2 = Dropout(0.5)(lstm_layer)

		# add dense layer
		dense_layer = Dense(1, 'sigmoid')(lstm_layer)

		# compile the model
		model = Model(in_layer, dense_layer)

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

		model.summary()

		history = model.fit(X_train, y_train, epochs=6, batch_size=1000, validation_data=(X_test, y_test))

		pred = model.predict(X_test)
		results = model.evaluate(X_test, y_test)
		labels = h.get_labels()

		report = classification_report(y_test, np.argmax(pred, axis=1), target_names=labels)
		print(report)
		h.write_report(report)
		h.plot_acc(history, X_train, y_train, op = 'normal')
		print('test_loss:', results[0], 'test_accuracy:', results[1])

		# get the embedding layer
		word_embeddings = model.get_layer('embedding').get_weights()[0]

		print('Shape of word_embeddings:', word_embeddings.shape)

		idx2word = re.get_idx2word(word2idx)
		print(DataFrame(word_embeddings, index=idx2word.values()).head(10))

		h.plot_tsne(word_embeddings, idx2word)

		return model