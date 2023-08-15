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
from inner_opt import Inner_opt

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
			return self.build_net

		# train without optimisation
		elif (args["train"] == 'none'):
			return self.build_net


	def build_net_opt(self, hp):

		# initialise sequential model
		self.model = Sequential()

		# create range and grid for optimisation
		hp_filters = hp.Choice('filters', values = [32, 64, 128, 256, 512])
		hp_units = hp.Choice('units', values = [32, 64, 128, 256, 512])
		hp_dropout = hp.Choice('rate', values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

		# build the model layer by layer
		# First layer
		self.add_first_layer(hp_filters)

		# Second layer
		self.add_convolution(hp_filters)
		self.add_pooling(hp_dropout)

		# Third layer
		self.add_convolution(hp_filters)

		# Fourth layer
		self.add_convolution(hp_filters)
		self.add_pooling(hp_dropout)

		# Fifth layer
		self.add_convolution(hp_filters)

		# Sixth layer
		self.add_convolution(hp_filters)
		self.add_pooling(hp_dropout)

		# add last layer
		self.add_last_layer(hp_units, hp_dropout)
		
		# compile with inner optimiser
		self.compile()

		return self.model


	def build_net(self):

		# initialise sequential model
		self.model = Sequential()

		# build the model layer by layer
		# First layer
		self.add_first_layer(32)

		# Second layer
		self.add_convolution(32)
		self.add_pooling(0.2)

		# Third layer
		self.add_convolution(64)

		# Fourth layer
		self.add_convolution(64)
		self.add_pooling(0.3)

		# Fifth layer
		self.add_convolution(128)

		# Sixth layer
		self.add_convolution(128)
		self.add_pooling(0.4)

		# add last layer
		self.add_last_layer(128, 0.5)

		# compile with inner optimiser
		self.compile()

		return self.model

	# function for the first layer
	def add_first_layer(self, filters):

		self.model.add(Conv2D(filters=filters, kernel_size=(5, 5), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		self.model.add(BatchNormalization(axis=-1))

	# function for the next layer
	def add_convolution(self, filters):

		self.model.add(Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(BatchNormalization(axis=-1))

	# function for pooling
	def add_pooling(self, dropout):

		self.model.add(MaxPooling2D((2, 2)))
		self.model.add(Dropout(rate=dropout))

	# function for the last layer
	def add_last_layer(self, units, dropout):

		self.model.add(Flatten())
		self.model.add(Dense(units=units, activation='relu', kernel_initializer='he_uniform'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(rate=dropout))
		self.model.add(Dense(3, activation='softmax'))

	# function to compile with inner optimiser
	def compile(self):

		inop = Inner_opt()

		# compile model
		print("[INFO] compiling model...")
		self.model.compile(loss="categorical_crossentropy", optimizer=inop.return_optimiser, metrics=["accuracy"])



