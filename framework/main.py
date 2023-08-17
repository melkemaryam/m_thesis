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

import os

if __name__ == '__main__':
	try:
		
		# create objects of all classes
		sk = Build_sk(None)
		bu = Build_tf(None)
		tu = Tuning(None)
		tr = Training()
		re = Read_data(None, None, None, None, None)
		prep = Preprocessing()
		pred = Predicting()
		ino = Inner_opt()
		he = Helper()
		a = Args()
		args = a.parse_arguments()

		# get data
		data = re.return_data()
		re.train_test_data(data)

		# predict images only with privided folder
		if(args["train"] == 'pred'):
			pred.prediction_process()

		# train, test, optimise, and predict with provided images
		elif((args["train"] == "all" or args["train"] == "one" or args["train"] == "none")):
			prep.prepare_data()

			if(args["model"] == 'basic' or args["model"] == 'cnn' or args["model"] == 'lstm' or args["model"] == 'bilstm'):
				tr.train_tf()

			elif(args["model"] == 'log' or args["model"] == 'svm' or args["model"] == 'nb'):
				tr.train_sk()

			pred.prediction_process()

	except KeyboardInterrupt:
		pass