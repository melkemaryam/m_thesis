# import packages
import argparse
from arguments import Args

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os
import glob
import collections

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from preprocessing import Preprocessing


class Read_data():

	def get_data(self):

		# get data
		data = self.return_data()
		print(data.head())

		#data = data[:1000]

		return data

	def train_test_data(self):

		a = Args()
		args = a.parse_arguments()

		data = self.get_data()

		if(args["model"] == 'basic' or args["model"] == 'cnn' or args["model"] == 'lstm' or args["model"] == 'bilstm'):
			# Features and Labels
			X = data['tokenised']
			y = pd.get_dummies(data['agg_label']).values

		elif(args["model"] == 'log' or args["model"] == 'svm' or args["model"] == 'nb'):
			# Features and Labels
			X = data['tokenised']
			y = data['agg_label']

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
		print(X_train[:10])

		return X_train, X_test, y_train, y_test

	def tokenise(self, max_len):

		# get values
		X_train, X_test, y_train, y_test = self.train_test_data()

		#preprocess 
		tokenizer = Tokenizer(num_words=10000, oov_token= "<OOV>")
		tokenizer.fit_on_texts(X_train)
		word_index = tokenizer.word_index
		X_train = tokenizer.texts_to_sequences(X_train)
		X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
		X_test = tokenizer.texts_to_sequences(X_test)
		X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')

		# convert lists into numpy arrays to make it work with TensorFlow 
		X_train = np.array(X_train)
		y_train = np.array(y_train)
		X_test = np.array(X_test)
		y_test = np.array(y_test)

		return X_train, X_test, y_train, y_test, tokenizer

	def adjust_data(self, data):

		data.fillna('', inplace=True)
		data.drop(columns = ['Unnamed: 0'], inplace=True)
		#data = data.iloc[:20000]

		return data

	def get_articles(self):

		a = Args()
		args = a.parse_arguments()

		# get titles only
		if (args["preprocess"] == 'no'):

			df_article = pd.read_csv("/Users/Hannah1/Downloads/articles_with_labels.csv", sep='\t', lineterminator='\n')
			data = self.adjust_data(df_article)

		elif (args["preprocess"] == 'yes'):

			p = Preprocessing()

			df_article = pd.read_csv("/Users/Hannah1/Downloads/articles_with_labels.csv", sep='\t', lineterminator='\n')
			data = self.adjust_data(df_article)
			data, path = p.apply_preprocessing(data)
			df_article = pd.read_csv(path, sep='\t', lineterminator='\n')
			data = self.adjust_data(df_article)

		return data

	def get_titles(self):
		
		a = Args()
		args = a.parse_arguments()

		# get titles only
		if (args["preprocess"] == 'no'):

			df_title = pd.read_csv("/Users/Hannah1/My Drive/Enigma/m_thesis/info/titles_with_labels.csv", sep='\t', lineterminator='\n')
			data = self.adjust_data(df_title)

		elif (args["preprocess"] == 'yes'):

			p = Preprocessing()

			df_title = pd.read_csv("/Users/Hannah1/My Drive/Enigma/m_thesis/info/titles_with_labels.csv", sep='\t', lineterminator='\n')
			data = self.adjust_data(df_title)
			data, path = p.apply_preprocessing(data)
			df_title = pd.read_csv(path, sep='\t', lineterminator='\n')
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

	def return_freq(self, text):

		# create a temporary list that includes all words
		freq_list = []
		word_list = []

		for row in text:

			#print(row)
			row = row.split()

			for word in row:
				#if word not in word_list:
				word_list.append(word)

		counter = collections.Counter(word_list)

		words = list(counter.keys())
		freq = list(counter.values())

		freqq = ['']*len(text)
		textt = pd.DataFrame()
		textt['freq'] = freqq
		c = 0

		for row in text:

			#print(row)
			row = row.split()
			freqs = []

			for word in row:

				if word in words:

					fr = counter.get(word)
					freqs.append(fr)

			#print(freqs)
			textt.loc[c, 'freq'] = freqs	

			c += 1

		textt = textt['freq'].to_list()
		#text.drop(columns='tokenised')

		return textt


	def prepare_data(self):

		text = self.get_data()

		# create a temporary list that includes all words
		word_list = []
		text = text['tokenised']
	    
		# create the empty dictionary
		word2idx = dict()

		# iterate through the entire corpus to create the list of words
		#for index, row in text.items():
		for row in text:

			#print(row)
			row = row.split()

			for word in row:
				if word not in word_list:
					word_list.append(word)

		# iterate through the list of words and add each word with the corresponding index to the dictionary
		for idx, word in enumerate(word_list):
			word2idx[word] = idx

		# return the final dictionary
		return word2idx

	def get_words(self, text):

		# create a temporary list that includes all words
		word_list = []
		text = text['tokenised']

		# iterate through the entire corpus to create the list of words
		#for index, row in text.items():
		for row in text:

			#print(row)
			row = row.split()

			for word in row:
				#if word not in word_list:
				word_list.append(word)

		return word_list

	def prepare_sent(self, text, word2idx):

		# create new list
		sents_as_ids = []
		text = text['tokenised']

		# iterate through entire corpus
		#for index, row in text.items():
		for row in text:

			#print(row)
			row = row.split()
			# create temporary list for the integers
			integer_list = []

			# iterate through all words in the sentence
			for word in row:

				# add the index of the word to the list of integers
				integer_list.append(word2idx[word])

			# add the list of integers to the final list
			sents_as_ids.append(integer_list)
    
		return sents_as_ids

	def get_idx2word(self, word2idx):

		idx2word = {index: word for word, index in word2idx.items()}

		return idx2word