"""

class: Helper

methods:
* write_report()
* write_score()
* set_model()
* check_path()
* plot_freq()
* plot_acc()
* plot_loss()


purpose:
* store helping functions used by other classes

"""

# import packages
import argparse
from arguments import Args
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
import os
from plot_keras_history import plot_history
from read_data import Read_data
import collections
import nltk
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

# Define a class named Helper
class Helper():

	# Define a method that captures all neccessary data and saves it in a markdown file
	def write_report(self, report):

		# create report
		file = open("../reports/report_" + datetime.now().strftime("%Y%m%d-%H%M") + ".md", "a") # path can be changed

		file.write(str(report))
		file.write("\n")
		file.close()

		print("[INFO] report written")

	# Define a method that saves the predictions of a model in separate markdown files depending on their label for future inspection
	def write_pred(self, label, report):

		# get arguments
		arg = Args()
		args = arg.parse_arguments()

		# save arguments as variables
		m_name = args["model"]

		if label == 0:

			# create report for negative predictions
			file = open("../predictions/neg_" + m_name + ".md", "a")

		elif label == 1:

			# create report for positive predictions
			file = open("../predictions/pos_" + m_name + ".md", "a")

		elif label == 2:

			# create report for neutral predictions
			file = open("../predictions/neu_" + m_name + ".md", "a")

		file.write(str(report))
		file.write("\n")
		file.write("\n")
		file.close()

		print("[INFO] report written")

	# print and save the training/testing scores
	def write_score(self, train_score, test_score):

		# get arguments
		arg = Args()
		args = arg.parse_arguments()

		# save arguments as variables
		m_name = args["model"]
		v_name = args["vector"]

		# print and save the received results
		print(f"\nShowing results for {v_name} and {m_name} Model")
		self.write_report(f"\nShowing results for {v_name} and {m_name} Model")
		print(f"Training Accuarcy: %.3f" % train_score)
		self.write_report(f"Training Accuarcy: %.3f" % train_score)
		print('Test Accuracy %.3f' % test_score)
		self.write_report('Test Accuracy %.3f' % test_score)

	# return the three lables for predictions
	def get_labels(self):
		
		# load labels
		labels = open("labels.csv").read().strip().split("\n")[1:]
		labels = [s.split(";")[1] for s in labels]

		return labels

	# plot the word frequency of the top 20 most frequently used words
	def plot_freq(self):

		re = Read_data()

		wordl = re.get_words()
		counter = collections.Counter(wordl)

		sns.set_style('darkgrid')
		words=nltk.FreqDist(wordl)
		ll = dict(sorted(counter.items(),key=lambda x:x[1], reverse=True))
		lls = ll.items()
		x,y = zip(*lls)

		plt.close()

		plt.plot(x[:20],y[:20])
		plt.title('Frequency Distribution')
		plt.xlabel('Words')
		plt.xticks(x[:20], rotation=90)
		plt.ylabel('Counts')
		plt.tight_layout()

		plt.savefig("../plots/plot_freq_" + datetime.now().strftime("%Y%m%d-%H%M"))
		self.write_report("![](../plots/plot_freq_" + datetime.now().strftime("%Y%m%d-%H%M")+ ".png)")
		
		return counter

	# plot the word similarities in a t-SNE plot
	def plot_tsne(self, word_embeddings, idx2word):

		tsne = TSNE(perplexity=3, n_components=2, init='pca', n_iter=5000, method='exact')
		np.set_printoptions(suppress=True)
		plot_only = 100 

		T = tsne.fit_transform(word_embeddings[plot_only:(plot_only*2), :])
		labels = [idx2word[i+1] for i in range(plot_only)]
		plt.figure(figsize=(14, 8))
		plt.scatter(T[:, 0], T[:, 1])
		for label, x, y in zip(labels, T[:, 0], T[:, 1]):
			plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points', ha='right', va='bottom')                      	                        

		plt.savefig("../plots/tsne_" + datetime.now().strftime("%Y%m%d-%H%M"))
		self.write_report("![](../plots/tsne_" + datetime.now().strftime("%Y%m%d-%H%M")+ ".png)")

	# plot the accuracy and loss of the models
	def plot_acc(self, model, v_train, y_train, op):

		arg = Args()
		args = arg.parse_arguments()

		# close previous plot
		plt.close()

		if(args["model"] == 'basic' or args["model"] == 'cnn' or args["model"] == 'lstm' or args["model"] == 'bilstm' or args["model"] == 'rnn'):

			plot_history(model.history)
			plt.title('Accuracy')

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

	# plot the loss of the models
	def plot_loss(self, model):

		arg = Args()
		args = arg.parse_arguments()

		# close previous plot
		plt.close()

		if(args["model"] == 'basic' or args["model"] == 'cnn' or args["model"] == 'lstm' or args["model"] == 'bilstm' or args["model"] == 'rnn'):

			plt.plot(model.history.history["loss"], label="training loss")
			plt.plot(model.history.history["val_loss"], label="validation loss")
			plt.title('Training Loss vs. Validation Loss ')
			plt.legend()
			plt.tight_layout()

			plt.savefig("../plots/plot_loss_" + datetime.now().strftime("%Y%m%d-%H%M"))
			self.write_report("![](../plots/plot_loss_" + datetime.now().strftime("%Y%m%d-%H%M")+ ".png)")
