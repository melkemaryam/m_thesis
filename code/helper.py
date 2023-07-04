# import packages
import argparse
from arguments import Args
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
import os

class Helper():

	def write_report(self, report):

		# create report
		file = open("../reports/report_" + datetime.now().strftime("%Y%m%d-%H%M") + ".md", "a")

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

	def plot_acc(self, model, v_train, y_train, op):

		arg = Args()
		args = arg.parse_arguments()

		# close previous plot
		plt.close()

		if (args["model"] == 'tf'):

			plt.plot(model.history.history["accuracy"], label="training accuracy")
			plt.plot(model.history.history["val_accuracy"], label="validation accuracy")
			plt.title('Training Accuracy vs. Validation accuracy ')

		else:

			LearningCurveDisplay.from_estimator(model, v_train, y_train)

		plt.legend()
		plt.tight_layout()

		# create the folder
		if (op == 'normal'):

			plt.savefig("../plots/plot_acc_" + datetime.now().strftime("%Y%m%d-%H%M"))
			self.write_report("![](../plots/plot_acc_" + datetime.now().strftime("%Y%m%d-%H%M")+ ".png)")
		
		else:

			plt.savefig("../plots/plot_acc_boost_" + datetime.now().strftime("%Y%m%d-%H%M"))
			self.write_report("![](../plots/plot_acc_boost_" + datetime.now().strftime("%Y%m%d-%H%M")+ ".png)")

		#plt.show()


	def plot_loss(self, model):

		arg = Args()
		args = arg.parse_arguments()

		# close previous plot
		plt.close()

		if (args["model"] == 'tf'):

			plt.plot(model.history.history["loss"], label="training loss")
			plt.plot(model.history.history["val_loss"], label="validation loss")
			plt.title('Training Loss vs. Validation Loss ')
			plt.legend()
			plt.tight_layout()

			plt.savefig("../plots/plot_loss_" + datetime.now().strftime("%Y%m%d-%H%M"))
			self.write_report("![](../plots/plot_loss_" + datetime.now().strftime("%Y%m%d-%H%M")+ ".png)")
			#plt.show()
