from arguments import Args
from build_model import Build_model
#from tuning import Tuning
#from training import Training
from read_data import Read_data
from preprocessing import Preprocessing
#from predicting import Predicting
#from helper import Helper
#from evaluating import Evaluating

import os

if __name__ == '__main__':
	try:

		# create objects of all classes
		bu = Build_model(None)
		#tu = Tuning(None)
		#tr = Training()
		re = Read_data()
		#prep = Preprocessing(None, None, None, None, None, None, None)
		#pred = Predicting()
		#he = Helper()
		#ev = Evaluating()
		a = Args()
		args = a.parse_arguments()

		# get titles only
		if (args["data"] == 'titles'):

			data = re.get_titles()

		# get articles only
		elif (args["data"] == 'articles'):

			data = re.get_articles()
			
		train_score, test_score, model, vector = bu.build_net(data)
		train_score, test_score, boosting = bu.boost_model(data, model)

	except KeyboardInterrupt:
		pass