from arguments import Args
from build_sk import Build_sk
from build_tf import Build_tf
from tuning import Tuning
from training import Training
from read_data import Read_data
from preprocessing import Preprocessing
from predicting import Predicting
from inner_opt import Inner_opt
from helper import Helper
# import packages
import argparse

import os

if __name__ == '__main__':
	try:
		
		# create objects of all classes
		sk = Build_sk(None)
		bu = Build_tf(None)
		tu = Tuning(None)
		tr = Training()
		re = Read_data()
		prep = Preprocessing()
		pred = Predicting()
		ino = Inner_opt()
		he = Helper()
		a = Args()
		args = a.parse_arguments()

		# predict images only with privided folder
		if(args["train"] == 'pred'):
			pred.prediction_process()

		# train, test, optimise, and predict with provided images
		elif((args["train"] == "all" or args["train"] == "one" or args["train"] == "none")):

			# get data
			#data = re.return_data()
			re.train_test_data()
			#print(data.head())

			if(args["model"] == 'basic' or args["model"] == 'cnn' or args["model"] == 'lstm' or args["model"] == 'bilstm'):
				tr.train_tf()

			elif(args["model"] == 'log' or args["model"] == 'svm' or args["model"] == 'nb'):
				tr.train_sk()

			pred.prediction_process()

	except KeyboardInterrupt:
		pass


# log/tfidf with training: python3 main.py -m log -v tfidf -d titles -tr none -pa ../output/log_tfidf.model
# log/count with training: python3 main.py -m log -v count -d titles -tr none -pa ../output/log_count.model
# log/w2v with training: python3 main.py -m log -v w2v -d titles -tr none -pa ../output/log_w2v.model

# svm/tfidf with training: python3 main.py -m svm -v tfidf -d titles -tr none -pa ../output/svm_tfidf.model
# svm/count with training: python3 main.py -m svm -v count -d titles -tr none -pa ../output/svm_count.model
# svm/w2v with training: python3 main.py -m svm -v w2v -d titles -tr none -pa ../output/svm_w2v.model

# nb/tfidf with training: python3 main.py -m nb -v tfidf -d titles -tr none -pa ../output/nb_tfidf.model
# nb/count with training: python3 main.py -m nb -v count -d titles -tr none -pa ../output/nb_count.model
# nb/w2v with training: python3 main.py -m nb -v w2v -d titles -tr none -pa ../output/nb_w2v.model

# basic with training: python3 main.py -m basic -d titles -tr none -pa ../output/basic.model








