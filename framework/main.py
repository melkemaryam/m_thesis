"""

methods:
* main

purpose:
* run the entire framework

"""

from arguments import Args
from build_sk import Build_sk
from build_tf import Build_tf
from training import Training
from read_data import Read_data
from preprocessing import Preprocessing
from predicting import Predicting
from inner_opt import Inner_opt
from helper import Helper

from hyperband import Hyper_band
from bayesian import Bayesian
# import packages
import argparse
from randoms import Randoms
from skip import Skip

import os

if __name__ == '__main__':
	try:
		
		# create objects of all classes
		sk = Build_sk(None)
		bu = Build_tf(None)
		tr = Training()
		re = Read_data()
		prep = Preprocessing()
		pred = Predicting()
		ino = Inner_opt()
		he = Helper()
		a = Args()
		args = a.parse_arguments()
		hb = Hyper_band()
		ba = Bayesian()
		rs = Randoms()
		ski = Skip()

		if(args["model"] == 'skip'):

			ski.create_train()

		# predicted labels with provided model
		if(args["train"] == 'pred'):
			pred.prediction_process()

		# train models without optimisation
		elif((args["train"] == "none")):

			if(args["model"] == 'basic' or args["model"] == 'cnn' or args["model"] == 'lstm' or args["model"] == 'bilstm' or args["model"] == 'rnn'):
				tr.train_tf()
				pred.prediction_process()

			elif(args["model"] == 'log' or args["model"] == 'svm' or args["model"] == 'nb'):
				tr.train_sk()
				pred.prediction_process()

		# train models with hyperparameter optimisation
		elif((args["train"] == "all")):
			
			if(args["optimiser"] == 'hyperband'):
				hb.main_train_net()

			elif(args["optimiser"] == 'bayesian'):
				ba.main_train_net()

			elif(args["optimiser"] == 'random'):
				rs.main_train_net()		

	except KeyboardInterrupt:
		pass


# Example Python commands:

# log/tfidf with training: python3 main.py -m log -v tfidf -d titles -tr none -pa ../output/log_tfidf.model
# log/count with training: python3 main.py -m log -v count -d titles -tr none -pa ../output/log_count.model
# log/w2v with training: python3 main.py -m log -v w2v -d titles -tr none -pa ../output/log_w2v.model

# svm/tfidf with training: python3 main.py -m svm -v tfidf -d titles -tr none -pa ../output/svm_tfidf.model
# svm/count with training: python3 main.py -m svm -v count -d titles -tr none -pa ../output/svm_count.model
# svm/w2v with training: python3 main.py -m svm -v w2v -d titles -tr none -pa ../output/svm_w2v.model

# nb/tfidf with training: python3 main.py -m nb -v tfidf -d titles -tr none -pa ../output/nb_tfidf.model
# nb/count with training: python3 main.py -m nb -v count -d titles -tr none -pa ../output/nb_count.model
# nb/w2v with training: python3 main.py -m nb -v w2v -d titles -tr none -pa ../output/nb_w2v.model

# basic with training: python3 main.py -m basic -inop adam -d titles -tr none -pa ../output/basic.model
# cnn with training: python3 main.py -m cnn -inop adam -d titles -tr none -pa ../output/cnn.model
# lstm with training: python3 main.py -m lstm -inop adam -d titles -tr none -pa ../output/lstm.model
# bilstm with training: python3 main.py -m bilstm -inop adam -d titles -tr none -pa ../output/bilstm.model
# rnn with training: python3 main.py -m rnn -inop adam -d titles -tr none -pa ../output/rnn.model

# rnn with optimisation: python3 main.py -m rnn -op hyperband -d titles -tr all -pa ../output/hyperband_rnn.model
# cnn with optimisation: python3 main.py -m cnn -op bayesian -d titles -tr all -pa ../output/bayesian_cnn.model
# lstm with optimisation: python3 main.py -m lstm -op random -d titles -tr all -pa ../output/random_lstm.model