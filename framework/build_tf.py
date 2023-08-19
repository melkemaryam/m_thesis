# import packages
import argparse
from arguments import Args

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.layers import Lambda, GlobalAveragePooling1D, Dense, Embedding, Conv1D, LSTM, Bidirectional, GlobalMaxPooling1D
from keras import backend as K
from inner_opt import Inner_opt
from read_data import Read_data

# arguments: choose between normal, lstm, cnn


class Build_tf():

	def __init__(self, model):

		self.model = model

	def return_model(self):

		arg = Args()
		args = arg.parse_arguments()

		# train with all optimisations
		if (args["train"] == 'all'):
			return self.build_net_opt

		# train with one optimisation
		elif (args["train"] == 'one'):
			return self.build_net()

		# train without optimisation
		elif (args["train"] == 'none'):
			return self.build_net()


	def build_net_opt(self, hp):

		# build the model layer by layer
		re = Read_data()
		a = Args()
		args = a.parse_arguments()

		# initialise sequential model
		self.model = Sequential()

		# create range and grid for optimisation
		hp_filters = hp.Choice('filters', values = [32, 64, 128, 256, 512])
		hp_units = hp.Choice('units', values = [32, 64, 128, 256, 512])
		hp_dropout = hp.Choice('rate', values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
		#hp_max_len = hp.Choice('input_length', values = [128, 256, 512, 1024])
		hp_embed_size = hp.Choice('output_dim', values = [64, 100, 150])

		MAX_LENGTH = 256
		VOCAB_SIZE = len(re.prepare_data())

		# build the model layer by layer
		# First layer
		self.add_embedding(VOCAB_SIZE, hp_embed_size, MAX_LENGTH, hp_dropout)

		# get titles only
		if(args["model"] == 'cnn'):

			self.add_first_conv(hp_filters)
			self.add_convolution(hp_filters)
			self.add_dropout(hp_dropout)
			self.add_convolution(hp_filters)
			self.add_convolution(hp_filters)
			self.add_dropout(hp_dropout)
			self.add_convolution(hp_filters)
			self.add_convolution(hp_filters)
			self.add_dropout(hp_dropout)
			self.add_last_cnn_layer(hp_units, hp_dropout)

		elif(args["model"] == 'lstm'):

			self.add_lstm(hp_embed_size, hp_dropout)
			self.add_last_layer(hp_units)

		elif(args["model"] == 'bilstm'):

			self.add_bilstm(hp_embed_size, hp_dropout)
			self.add_last_layer(hp_units)

		elif(args["model"] == 'rnn'):

			self.add_rnn(hp_embed_size, hp_dropout)
			self.add_last_cnn_layer(hp_units, hp_dropout)

		elif(args["model"] == 'basic'):

			self.add_pooling(hp_dropout)
			self.add_last_layer(hp_units)

		# compile with inner optimiser
		self.compile()

		return self.model


	def build_net(self):

		re = Read_data()
		a = Args()
		args = a.parse_arguments()

		MAX_LENGTH = 256 # or 200
		EMBED_SIZE = 100 # or 64
		VOCAB_SIZE = len(re.prepare_data())

		# initialise sequential model
		self.model = Sequential()

		# build the model layer by layer
		# First layer
		self.add_embedding(VOCAB_SIZE, EMBED_SIZE, MAX_LENGTH, 0.2)

		# get titles only
		if(args["model"] == 'cnn'):

			self.add_first_conv(32)
			self.add_convolution(32)
			self.add_dropout(0.1)
			self.add_convolution(64)
			self.add_convolution(64)
			self.add_dropout(0.2)
			self.add_convolution(128)
			self.add_convolution(128)
			self.add_dropout(0.5)
			self.add_convolution(128)
			self.add_convolution(128)
			self.add_dropout(0.5)
			self.add_last_cnn_layer(24, 0.3)

		elif(args["model"] == 'lstm'):

			self.add_lstm(EMBED_SIZE, 0.5)
			self.add_last_cnn_layer(24, 0.5)

		elif(args["model"] == 'bilstm'):

			self.add_bilstm(EMBED_SIZE, 0.5)
			self.add_last_cnn_layer(24, 0.5)

		elif(args["model"] == 'rnn'):

			self.add_rnn(EMBED_SIZE, 0.5)
			self.add_last_cnn_layer(24, 0.5)

		elif(args["model"] == 'basic'):

			#self.model.add(GlobalAveragePooling1D())
			self.add_dropout(0.2)
			self.add_last_cnn_layer(24, 0.5)

		# compile with inner optimiser
		self.compile()

		return self.model

	def add_embedding(self, VOCAB_SIZE, EMBED_SIZE, MAX_LENGTH, dropout):

		a = Args()
		args = a.parse_arguments()

		# FOR BILSTM WITHOUT MASK ZERO
		if(args["model"] == 'bilstm' or args["model"] == 'rnn'):
			self.model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, mask_zero=False, input_length = MAX_LENGTH))
		else:
			self.model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, mask_zero=True, input_length = MAX_LENGTH))
		
		self.model.add(Dropout(rate=dropout))

	# function for the first layer
	def add_first_conv(self, filters):

		self.model.add(Conv1D(filters=filters, kernel_size=(5), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(BatchNormalization(axis=-1))

	# function for the next layer
	def add_convolution(self, filters):

		self.model.add(Conv1D(filters=filters, kernel_size=(3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(BatchNormalization(axis=-1))

	# function for pooling
	def add_dropout(self, dropout):

		self.model.add(Dropout(rate=dropout))

	def add_lstm(self, EMBED_SIZE, dropout):

		self.model.add(LSTM(EMBED_SIZE))
		self.model.add(Dropout(rate=dropout))

	def add_bilstm(self, EMBED_SIZE, dropout):

		self.model.add(Bidirectional(LSTM(EMBED_SIZE)))
		self.model.add(Dropout(rate=dropout))

	def add_rnn(self, EMBED_SIZE, dropout):

		self.model.add(Bidirectional(LSTM(EMBED_SIZE, return_sequences=True)))
		self.model.add(Bidirectional(LSTM(50)))
		self.model.add(Dropout(rate=dropout))

	def add_last_layer(self, units):

		self.model.add(Dense(units=units, activation='relu', kernel_initializer='he_uniform'))
		self.model.add(Dense(3, activation='softmax'))
		#self.model.add(Dense(1, "sigmoid"))

	# function for the last layer
	def add_last_cnn_layer(self, units, dropout):

		#self.model.add(GlobalMaxPooling1D())
		self.model.add(Flatten())
		self.model.add(Dense(units=units, activation='relu', kernel_initializer='he_uniform'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(rate=dropout))
		self.model.add(Dense(3, activation='softmax'))
		#self.model.add(Dense(1, "sigmoid"))

	# function to compile with inner optimiser
	def compile(self):

		inop = Inner_opt()
		op = inop.return_optimiser()

		# compile model
		print("[INFO] compiling model...")
		self.model.compile(loss="categorical_crossentropy", optimizer=op, metrics=["accuracy"])