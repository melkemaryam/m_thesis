"""

class: Args

methods:
* parse_arguments()

purpose:
* get arguments from function call

"""

# Import the argparse module for command-line argument parsing
import argparse

# Define a class named Args
class Args():

	# Define a method named parse_arguments() for getting arguments from function call
	def parse_arguments(self):
		
		# Create an ArgumentParser object
		ap = argparse.ArgumentParser()

		# Add arguments to the argument parser
		ap.add_argument("-m", "--model", default='rnn', choices=['log', 'svm', 'nb', 'basic', 'cnn', 'lstm', 'bilstm', 'rnn', 'bert', 'skip'], required=False, help="choose model")
		ap.add_argument("-v", "--vector", default='tfidf', choices=['tfidf', 'count', 'w2v'], required=False, help="choose vectoriser")
		ap.add_argument("-op", "--optimiser", default='hyperband', choices=['bayesian', 'hyperband', 'random'], required=False, help="optimisation method for classifier")
		ap.add_argument("-inop", "--inner_optimiser", default='adam', choices=['adam', 'sgd', 'rms'], required=False, help="inner optimisation method for classifier")
		ap.add_argument("-d", "--data", default='titles', choices=['titles', 'articles'], required=False, help="titles or articles")
		ap.add_argument("-pre", "--preprocess", default='no', choices=['yes', 'no'], required=False, help="choose whether data needs to be preprocessed first")
		ap.add_argument("-tr", "--train", default='pred', choices=['all', 'none', 'pred'], required=False, help="give info on whether to optimise, 'pred' = no training")
		ap.add_argument("-pa", "--path", default='../output/rnn.model', required=False, help="output path to model")	
		
		# Parse the command-line arguments and store them in a dictionary
		args = vars(ap.parse_args())

		# Return the dictionary containing the parsed arguments
		return args