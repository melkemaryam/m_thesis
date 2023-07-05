# import packages
import argparse
from arguments import Args

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os
import glob

from sklearn.model_selection import train_test_split


class Read_data():

	def train_test_data(self, data):

		# Features and Labels
		X = data['tokenised']
		y = data['agg_label']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

		return X_train, X_test, y_train, y_test

	def adjust_data(self, data):

		data.fillna('', inplace=True)
		data.drop(columns = ['Unnamed: 0'], inplace=True)
		data = data.iloc[:10000]

		return data

	def get_articles(self):

		df_article = pd.read_csv("/Users/Hannah1/Downloads/article_df_final.csv", sep='\t', lineterminator='\n')
		data = self.adjust_data(df_article)

		return data

	def get_titles(self):

		df_title = pd.read_csv("/Users/Hannah1/Downloads/title_df_final.csv", sep='\t', lineterminator='\n')
		data = self.adjust_data(df_title)

		return data

	def return_data(self):

		a = Args()
		args = a.parse_arguments()

		# get titles only
		if (args["data"] == 'titles'):

			data = self.get_titles()

			return data

		# get articles only
		elif (args["data"] == 'articles'):

			data = self.get_articles()

			return data