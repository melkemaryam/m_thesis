"""

class: Bayesian

methods:
* main_train_net()
* build_net()
* add_embedding()
* add_first_conv()
* add_convolution()
* add_dropout()
* add_lstm()
* add_bilstm()
* add_last_layer()
* add_last_cnn_layer()
* tuning()
* training()
* write_report()

purpose:
* use bayesian hyperparameter optimisation

"""

# set the matplotlib backend so figures can be saved in the background
import matplotlib

# import packages
import argparse
from datetime import datetime
import numpy as np
import os
import random

import kerastuner
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import MSE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from keras.layers import Lambda, GlobalAveragePooling1D, Dense, Embedding, Conv1D, LSTM, Bidirectional, GlobalMaxPooling1D

import matplotlib.pyplot as plt
from arguments import Args
import time

from read_data import Read_data
from helper import Helper
from predicting import Predicting

# Define a class named Bayesian
class Bayesian():

	# Define the main_train_net method
	def main_train_net(self):

		# Create an instance of the Args class
		arg = Args()
		self.args = arg.parse_arguments()

		# Create an instance of the Bayesian class
		self.net = Bayesian()

		# Prepare the data and the model
		self.tuning()
		history = self.training()

		# Perform label prediction
		pr = Predicting()
		pr.prediction_process()

	# Define the build_net method with hyperparameter tuning
	def build_net(self, hp):

		# Create an instance of the Read_data class
		re = Read_data()

		# Create an instance of the Args class
		a = Args()
		args = a.parse_arguments()

		# Initialize a sequential model
		self.model = Sequential()

		# Define hyperparameters for tuning
		hp_filters = hp.Choice('filters', values = [32, 64, 128, 256, 512])
		hp_units = hp.Choice('units', values = [32, 64, 128, 256, 512])
		hp_dropout = hp.Choice('rate', values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
		hp_embed_size = hp.Choice('output_dim', values = [64, 100, 150])

		MAX_LENGTH = 256
		VOCAB_SIZE = len(re.prepare_data())

		# Create the embedding layer
		self.add_embedding(VOCAB_SIZE, hp_embed_size, MAX_LENGTH, hp_dropout)

		# Build the model layer by layer based on the selected model type
		if(args["model"] == 'cnn'):

			# Add convolutional layers
			self.add_first_conv(hp_filters)
			self.add_convolution(hp_filters)
			self.add_dropout(hp_dropout)
			self.add_convolution(hp_filters)
			self.add_convolution(hp_filters)
			self.add_dropout(hp_dropout)
			self.add_convolution(hp_filters)
			self.add_convolution(hp_filters)
			self.add_dropout(hp_dropout)
			self.add_last_layer(hp_units, hp_dropout)

		elif(args["model"] == 'lstm'):

			# Add an LSTM layer
			self.add_lstm(hp_embed_size, hp_dropout)
			self.add_last_layer(hp_units)

		elif(args["model"] == 'bilstm'):

			# Add a bidirectional LSTM layer
			self.add_bilstm(hp_embed_size, hp_dropout)
			self.add_last_layer(hp_units)

		elif(args["model"] == 'rnn'):

			# Add an RNN layer
			self.add_rnn(hp_embed_size, hp_dropout)
			self.add_last_ayer(hp_units, hp_dropout)

		elif(args["model"] == 'basic'):

			# Keep the model basic
			self.add_dropout(hp_dropout)
			self.add_last_layer(hp_units)

		# Compile the model with an inner optimizer
		print("[INFO] compiling model...")
		
		hp_learning_rate = hp.Choice('learning_rate', values = [1e-1, 1e-2, 1e-3, 1e-4])
		opt = Adam(learning_rate=hp_learning_rate)

		self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

		return self.model

	# Define the add_embedding method
	def add_embedding(self, VOCAB_SIZE, EMBED_SIZE, MAX_LENGTH, dropout):

		a = Args()
		args = a.parse_arguments()

		# FOR BILSTM AND RNN WITHOUT MASK ZERO
		if(args["model"] == 'bilstm' or args["model"] == 'rnn'):
			self.model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, mask_zero=False, input_length = MAX_LENGTH))
		else:
			self.model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, mask_zero=True, input_length = MAX_LENGTH))
		
		self.model.add(Dropout(rate=dropout))

	# Add the first convolutional layer
	def add_first_conv(self, filters):

		self.model.add(Conv1D(filters=filters, kernel_size=(5), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(BatchNormalization(axis=-1))

	# function for the next convolutional layer
	def add_convolution(self, filters):

		self.model.add(Conv1D(filters=filters, kernel_size=(3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(BatchNormalization(axis=-1))

	# Define the add_dropout method
	def add_dropout(self, dropout):

		self.model.add(Dropout(rate=dropout))

	# Define the add_lstm method
	def add_lstm(self, EMBED_SIZE, dropout):

		self.model.add(LSTM(EMBED_SIZE))
		self.model.add(Dropout(rate=dropout))

	# Define the add_bilstm method
	def add_bilstm(self, EMBED_SIZE, dropout):

		self.model.add(Bidirectional(LSTM(EMBED_SIZE)))
		self.model.add(Dropout(rate=dropout))

	# Define the add_rnn method
	def add_rnn(self, EMBED_SIZE, dropout):

		half = int(EMBED_SIZE/2)

		self.model.add(Bidirectional(LSTM(EMBED_SIZE, return_sequences=True)))
		self.model.add(Bidirectional(LSTM(half)))
		self.model.add(Dropout(rate=dropout))

	# Define the add_last_layer method
	def add_last_layer(self, units, dropout):

		# Add the last dense and output layers with softmax
		self.model.add(Flatten())
		self.model.add(Dense(units=units, activation='relu', kernel_initializer='he_uniform'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(rate=dropout))
		self.model.add(Dense(3, activation='softmax'))

	# Define the tuning method for hyperparameter optimization
	def tuning(self):

		re = Read_data()
		h = Helper()

		# Set up hyperparameter tuning with Bayesian Optimization
		self.tuner = BayesianOptimization(
			self.build_net,
			objective='accuracy',
			max_trials=10,
			executions_per_trial=1,
			directory='bayesian_data') #change the directory name here  when rerunning the cell else it gives "Oracle exit error" 

		self.write_report(self.tuner.search_space_summary())

		MAX_LENGTH = 256

		X_train, X_test, y_train, y_test, tok = re.tokenise(MAX_LENGTH)

		# tune the hyperparameters
		self.tuner.search(X_train, y_train,
			epochs=10,
			validation_data=(X_test, y_test))

		self.write_report(self.tuner.results_summary())

		# get best parameters
		self.best_hyperparameters = self.tuner.get_best_hyperparameters(1)[0]
		print(self.best_hyperparameters.values)
		self.write_report(self.best_hyperparameters.values)

		for data in self.tuner.get_best_hyperparameters(1):
			print(data.values)
			self.write_report(data.values)

		n_best_models = self.tuner.get_best_models(num_models=2)
		print(n_best_models[0].summary()) # best-model summary
		self.write_report(n_best_models[0].summary())

	# Define the training method
	def training(self):

		# train the network
		print("[INFO] training network...")
		MAX_LENGTH = 256

		re = Read_data()
		h = Helper()
		a = Args()
		args = a.parse_arguments()

		X_train, X_test, y_train, y_test, tok = re.tokenise(MAX_LENGTH)

		log_dir = "../logs/fit_bo/" + datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
		callback = EarlyStopping(monitor='val_loss', patience=3)

		self.train = self.tuner.hypermodel.build(self.best_hyperparameters)
		self.train.summary()

		# Train the network and evaluate its performance
		history = self.train.fit(X_train,
			y_train,
			validation_data=(X_test, y_test),
			epochs=200,
			batch_size=512, 
			callbacks=[callback, tensorboard],
			verbose=1)

		# save the network to disk
		print("[INFO] serializing network to '{}'...".format(args["path"]))
		self.train.save(args["path"])

		# evaluate the network
		print("[INFO] evaluating network...")

		# get labels
		labels = h.get_labels()

		# test model
		predictions = self.train.predict(X_test)
		results = self.train.evaluate(X_test, y_test)
		report = classification_report(y_test.argmax(axis=1), np.argmax(predictions, axis=1), target_names=labels)

		print(report)
		self.write_report(report)

		_, acc = self.train.evaluate(X_test, y_test, verbose=0)
		print('> %.3f' % (acc * 100.0))
		self.write_report('> %.3f' % (acc * 100.0))
		h.plot_acc(history, X_train, y_train, op = 'normal')
		print('test_loss:', results[0], 'test_accuracy:', results[1])
		h.write_score(acc, results[1])

	# Define the write_report method
	def write_report(self, report):

		file = open("../reports/bayesian/test_report_bo_adam_" + datetime.now().strftime("%Y%m%d-%H%M") + ".md", "a")

		file.write(str(report))
		file.write("\n")
		file.close()

		print("[INFO] report written")