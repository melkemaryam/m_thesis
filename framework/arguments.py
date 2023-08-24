"""

class: Args

methods:
* parse_arguments()

purpose:
* get arguments from function call

"""

import argparse

class Args():

	def parse_arguments(self):
		# create argument parser
		ap = argparse.ArgumentParser()
		ap.add_argument("-m", "--model", default='lstm', choices=['log', 'svm', 'nb', 'basic', 'cnn', 'lstm', 'bilstm', 'rnn', 'bert', 'skip'], required=False, help="choose model")
		ap.add_argument("-v", "--vector", default='tfidf', choices=['tfidf', 'count', 'w2v'], required=False, help="choose vectoriser")
		ap.add_argument("-op", "--optimiser", default='hyperband', choices=['bayesian', 'hyperband', 'random'], required=False, help="optimisation method for classifier")
		ap.add_argument("-inop", "--inner_optimiser", default='adam', choices=['adam', 'sgd', 'rms'], required=False, help="inner optimisation method for classifier")
		ap.add_argument("-d", "--data", default='titles', choices=['titles', 'articles'], required=False, help="titles or articles")
		ap.add_argument("-pre", "--preprocess", default='no', choices=['yes', 'no'], required=False, help="choose whether data needs to be preprocessed first")
		ap.add_argument("-tr", "--train", default='pred', choices=['all', 'none', 'pred'], required=False, help="give info on whether to optimise, 'pred' = no training")
		ap.add_argument("-pa", "--path", default='../output/new.model', required=False, help="output path to model")	
		args = vars(ap.parse_args())

		return args