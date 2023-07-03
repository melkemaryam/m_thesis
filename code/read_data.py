# import packages
import argparse
from arguments import Args

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os
import glob

class Read_data():

	def get_articles(self):

		df_article = pd.read_csv("/Users/Hannah1/Downloads/article_df_final.csv", sep='\t', lineterminator='\n')
		df_article.fillna('', inplace=True)
		df_article.drop(columns = ['Unnamed: 0'], inplace=True)
		df_article = df_article.iloc[:10000]

		return df_article

	def get_titles(self):

		df_title = df_title = pd.read_csv("/Users/Hannah1/Downloads/title_df_final.csv", sep='\t', lineterminator='\n')
		df_title.fillna('', inplace=True)
		df_title.drop(columns = ['Unnamed: 0'], inplace=True)
		df_title = df_title.iloc[:10000]

		return df_title