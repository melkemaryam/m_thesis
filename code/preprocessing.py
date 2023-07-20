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

	def pre_process(self, data):

	    publishers = ['The New York Times', 'Breitbart', 'CNN', 'Business Insider', 'Fox News', 'Talking Points Memo', 'Buzzfeed News', 'National Review', 'New York Post', 'The Guardian', 'NPR', 'Reuters', 'Vox', 'Washington Post', 'Associated Press']

	    text = df

	    for a in publishers:
	      if a in df:
	        text = df.replace(a, '')

	    # take care of punction
	    text = re.sub(r"([.,;:!?'\"“\(\)])(\w)", r"\1 \2", text) # when at the beginning of a string, separate punctuation
	    text = re.sub(r"(\w)([.,;:!?'\"”\)])", r"\1 \2", text) # when at the end of a string, separate punctuation

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
	    x = '[{}]'.format(re.escape(string.punctuation)+'…').replace("...", "").replace("-", "")
	    pattern = re.compile(x)
	    tokens = [f for f in filter(None, [pattern.sub('', token) for token in tokens])]

	    # remove stop words
	    stopwords = nltk.corpus.stopwords.words('english')
	    tokens = [token for token in tokens if token not in stopwords]

	    # apply lemmatisation
	    lemmatiser = WordNetLemmatizer()
	    tokens = [lemmatiser.lemmatize(token) for token in tokens]


	    # return final list of tokens
	    return tokens

	def prepare_data(self, text):

		# create a temporary list that includes all words
	    word_list = []
	    
	    # create the empty dictionary
	    word2idx = dict()
	    
	    # iterate through the entire corpus to create the list of words
	    for sentence in text:
	        for word in sentence:
	            if word not in word_list:
	                word_list.append(word)
	    
	    # iterate through the list of words and add each word with the corresponding index to the dictionary
	    for idx, word in enumerate(word_list):
	        word2idx[word] = idx
	   
	    # return the final dictionary
	    return word2idx

	def prepare_sid(self, text):
    
	    # create new list
	    sents_as_ids = []
	    
	    # iterate through entire corpus
	    for sentence in text:
	        
	        # create temporary list for the integers
	        integer_list = []
	        
	        # iterate through all words in the sentence
	        for word in sentence:
	            
	            # add the index of the word to the list of integers
	            integer_list.append(word2idx[word])
	        
	        # add the list of integers to the final list
	        sents_as_ids.append(integer_list)
	            
	    return sents_as_ids

	def get_idx2word(self):
    
	    idx2word = {index: word for word, index in word2idx.items()}
	    
	    return idx2word













