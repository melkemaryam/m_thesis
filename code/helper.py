# import packages
import argparse
from arguments import Args
from datetime import datetime

class Helper():

	def write_report(self, report):

		# create report
		file = open("../reports/report_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt", "a")

		file.write(str(report))
		file.write("\n")
		file.close()

		print("[INFO] report written")

	def check_path(self):
		
		# create the folder
		if not os.path.exists('Predict'):
			os.makedirs('Predict')

		# empty Predict directory, so only new data is shown
		files = glob.glob('Predict/*')
		for f in files:
			os.remove(f)

	def get_labels(self):
		
		# load sign names
		labels = open("labels.csv").read().strip().split("\n")[1:]
		labels = [s.split(";")[1] for s in labels]

		return labels