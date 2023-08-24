"""

class: Preprocessing

methods:
* pre_process()
* apply_preprocess()

purpose:
* pre-process given data

"""

# import packages
import argparse
from arguments import Args
import numpy as np
import os
import re
import nltk
from spacy.lang.en.stop_words import STOP_WORDS
import spacy

from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import string
from string import punctuation
from nltk.corpus import stopwords

import nltk.tokenize
punc = string.punctuation
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer  # lemmatization
nltk.download('wordnet')


class Preprocessing():

	def pre_process(self, df):

		publishers = ['The New York Times', 'Breitbart', 'CNN', 'Business Insider', 'Fox News', 'Talking Points Memo', 'Buzzfeed News', 'National Review', 'New York Post', 'The Guardian', 'NPR', 'Reuters', 'Vox', 'Washington Post', 'Associated Press']

		text = df

		for a in publishers:
			if a in df:
				text = df.replace(a, '')

		# take care of punction
		text = re.sub(r"([.,;‘:’!-?'\"“\(\)])(\w)", r"\1 \2", text) # when at the beginning of a string, separate punctuation
		text = re.sub(r"(\w)([.,;‘:’!-?'\"”\)])", r"\1 \2", text) # when at the end of a string, separate punctuation
		text = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', text)

		# remove any numbers
		numbers = r'\d+'
		text = re.sub(pattern=numbers, repl=" ", string=text)

		# split the string into separate tokens
		tokens = re.split(r"\s+",text)

		# normalise all words into lowercase
		final = [t.lower() for t in tokens]

		tokens = []

		# remove any strings signalising end of line
		for i in range(len(final)):
			if final[i] != '-' and final[i] != '—' and final[i] != '“' and final[i] != '”':
				tokens.append(final[i])

		# remove any unncessary punctuation connected to words
		x = '[{}]'.format(re.escape(string.punctuation)+'…').replace("...", "").replace("-", "").replace("‘", "").replace("’", "")
		pattern = re.compile(x)
		tokens = [f for f in filter(None, [pattern.sub('', token) for token in tokens])]

		# remove stop words
		stopwords = nltk.corpus.stopwords.words('english')
		tokens = [token for token in tokens if token not in stopwords]

		# apply lemmatisation
		lemmatiser = WordNetLemmatizer()
		tokens = [lemmatiser.lemmatize(token) for token in tokens]

		tokens = [w for w in tokens if len(w)>1]

		# return final list of tokens
		return tokens

	def test_pp(self):

		text_test = "I have e ' h , the — high ground ” , “ Anakin! - The New York Times"
		print(text_test)
		tokens_test = self.pre_process(text_test)
		print(tokens_test)

	def apply_preprocess(self, data):

		arg = Args()
		args = arg.parse_arguments()

		print("[INFO] preprocessing the data...")

		if 'tokenised' in data.columns:

			data.rename(columns = {'tokenised':'old_tokens'}, inplace = True)
			data.drop(columns = ["old_tokens"], inplace=True)

		tokenised = ['']* int(data.shape[0])
		data['tokenised'] = tokenised

		for index, row in data.iterrows():
			token = []

			if (args["data"] == 'titles'):
				token = pre_process(row['title'])
			elif (args["data"] == 'articles'):
				token = pre_process(row['article'])
			
			# Using .join() to Convert a List to a String
			conv_token = ' '.join(token)

			data.loc[index, 'tokenised'] = conv_token

		data.head()

		path = self.save_data(data)

		print("[INFO] data fully preprocessed...")

		return data, path

	def save_data(self, data):

		arg = Args()
		args = arg.parse_arguments()

		if (args["data"] == 'titles'):
			
			# CHANGE PATH EVERY TIME
			path = "/Users/Hannah1/My Drive/Enigma/m_thesis/info/title_df_final2.csv"
			data.to_csv("/Users/Hannah1/My Drive/Enigma/m_thesis/info/title_df_final2.csv", sep='\t')

			return path

		elif (args["data"] == 'articles'):

			# CHANGE PATH EVERY TIME
			path = "/Users/Hannah1/Downloads/article_df_final2.csv"
			data.to_csv("/Users/Hannah1/Downloads/article_df_final2.csv", sep='\t')

			return path