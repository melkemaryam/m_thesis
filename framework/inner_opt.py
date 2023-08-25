"""

class: Inner_opt

methods:
* return_optimiser()

purpose:
* get correct optimiser

"""

# import packages
import argparse
from arguments import Args

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# Define a class named Inner_opt
class Inner_opt():

	def return_optimiser(self):

		arg = Args()
		args = arg.parse_arguments()

		# return the correct inner optimiser
		if args["inner_optimiser"] == 'adam' and args["train"] == 'none':
			return self.direct_adam()
		elif args["inner_optimiser"] == 'sgd' and args["train"] == 'none':
			return self.direct_sgd()
		elif args["inner_optimiser"] == 'rms' and args["train"] == 'none':
			return self.direct_rms()

	def direct_adam(self):

		# initialise the optimiser
		optimiser = Adam(learning_rate=0.00001, clipnorm=1.)

		return optimiser

	def direct_sgd(self):

		# initialise the optimiser
		optimiser = SGD(learning_rate=0.0001)

		return optimiser

	def direct_rms(self):

		# initialise the optimiser
		optimiser = RMSprop(learning_rate=0.001)

		return optimiser