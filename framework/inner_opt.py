# import packages
import argparse
from arguments import Args

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorboard.plugins.hparams import api as hp

class Inner_opt():

	def return_optimiser(self):

		arg = Args()
		args = arg.parse_arguments()

		# return the correct inner optimiser
		if args["inner_optimiser"] == 'adam' and (args["train"] == 'all' or args["train"] == 'one'):
			return self.optimise_adam()
		elif args["inner_optimiser"] == 'sgd' and (args["train"] == 'all' or args["train"] == 'one'):
			return self.optimise_sgd
		elif args["inner_optimiser"] == 'rms' and (args["train"] == 'all' or args["train"] == 'one'):
			return self.optimise_rms
		elif args["inner_optimiser"] == 'adam' and args["train"] == 'none':
			return self.direct_adam()
		elif args["inner_optimiser"] == 'sgd' and args["train"] == 'none':
			return self.direct_sgd()
		elif args["inner_optimiser"] == 'rms' and args["train"] == 'none':
			return self.direct_rms()


	def optimise_adam(self, hp):

		# initialise hyperarameter space
		hp_learning_rate = hp.choice('learning_rate', values = [1e-1, 1e-2, 1e-3, 1e-4])

		# initialise the optimiser
		optimiser = Adam(learning_rate=hp_learning_rate)

		return optimiser

	def direct_adam(self):

		# initialise the optimiser
		optimiser = Adam(learning_rate=0.001)

		return optimiser

	def optimise_sgd(self, hp):

		# initialise hyperarameter space
		hp_learning_rate = hp.Choice('learning_rate', values = [1e-1, 1e-2, 1e-3, 1e-4])

		# initialise the optimiser
		optimiser = SGD(learning_rate=hp_learning_rate)

		return optimiser

	def direct_sgd(self):

		# initialise the optimiser
		optimiser = SGD(learning_rate=0.001)

		return optimiser

	def optimise_rms(self, hp):

		# initialise hyperarameter space
		hp_learning_rate = hp.Choice('learning_rate', values = [1e-1, 1e-2, 1e-3, 1e-4])

		# initialise the optimiser
		optimiser = RMSprop(learning_rate=hp_learning_rate)

		return optimiser

	def direct_rms(self):

		# initialise the optimiser
		optimiser = RMSprop(learning_rate=0.001)

		return optimiser