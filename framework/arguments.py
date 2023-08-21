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
		ap.add_argument("-m", "--model", default='lstm', choices=['log', 'svm', 'nb', 'basic', 'cnn', 'lstm', 'bilstm', 'rnn', 'bert'], required=False, help="choose model")
		ap.add_argument("-v", "--vector", default='tfidf', choices=['tfidf', 'count', 'w2v'], required=False, help="choose vectoriser")
		ap.add_argument("-op", "--optimiser", default='hyperband', choices=['boost', 'bayesian', 'hyperband', 'random'], required=False, help="optimisation method for classifier")
		ap.add_argument("-inop", "--inner_optimiser", default='adam', choices=['adam', 'sgd', 'rms'], required=False, help="inner optimisation method for classifier")
		ap.add_argument("-d", "--data", default='titles', choices=['titles', 'articles'], required=False, help="titles or articles")
		ap.add_argument("-pre", "--preprocess", default='no', choices=['yes', 'no'], required=False, help="choose whether data needs to be preprocessed first")
		ap.add_argument("-pr", "--predictions", default='Predict',required=False, help="path to the directory with images to predict or path to file to create new images")
		ap.add_argument("-tr", "--train", default='pred', choices=['all', 'one', 'none', 'pred'], required=False, help="give info on how many parameters to optimise, 'pred' = no training")
		ap.add_argument("-pa", "--path", default='../output/basic.model', required=False, help="path to model")	
		args = vars(ap.parse_args())

		return args