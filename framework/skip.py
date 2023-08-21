# import packages
import argparse
from arguments import Args

# pandas and numpy
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from pandas import DataFrame


#import libraries and modules
import io

from keras.preprocessing.sequence import skipgrams
from keras.layers import Dot, Embedding, Input
from keras.layers.core import Dense, Reshape
from keras.models import Model
from keras.utils.vis_utils import plot_model

from read_data import Read_data
from IPython.display import SVG, display
from keras.utils import vis_utils
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import nltk

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from helper import Helper
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
# good tutorial: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

class Skip():

	def create_train(self):

		re = Read_data()
		h = Helper()

		word2idx = re.prepare_data()
		idx2word = re.get_idx2word(word2idx)

		h.plot_freq()
		data = re.get_data()

		h.write_report(f"The size of this dataset is %.1f" % len(data))
		print('Number of unique words:', len(word2idx))
		h.write_report(('Number of unique words: ' + str(len(word2idx))))
		print('\nSample word2idx: ', list(word2idx.items())[:10])
		h.write_report(('\nSample word2idx: ' + str(list(word2idx.items())[:10])))
		print('\nSample idx2word:', list(idx2word.items())[:10])
		h.write_report(('\nSample idx2word: ' + str(list(idx2word.items())[:10])))
		print('\nSample sents_as_id:', re.prepare_sent(data[:10], word2idx))
		h.write_report(('\nSample sents_as_id: ' + str(re.prepare_sent(data[:10], word2idx))))

		VOCAB_SIZE = len(word2idx)
		EMBED_SIZE = 100 # We are creating 100D embeddings.

		skip_grams = []

		for sequence in re.prepare_sent(data, word2idx):

			skip_grams.append(skipgrams(sequence, VOCAB_SIZE, shuffle=False))

		pairs, labels = skip_grams[0][0], skip_grams[0][1]

		# The input is an array of target indices e.g. [2, 45, 7, 23,...9]
		target_word = Input((1,), dtype='int32')

		# feed the words into the model using the Keras <Embedding> layer. This is the hidden layer 
		# from whose weights we will get the word embeddings.
		target_embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, name='target_embed_layer',
									embeddings_initializer='glorot_uniform',
									input_length=1)(target_word)

		# at this point, the input would of the shape (num_inputs x 1 x embed_size) and has to be flattened 
		# or reshaped into a (num_inputs x embed_size) tensor.
		target_input = Reshape((EMBED_SIZE, ))(target_embedding)

		# The input is an array of target indices e.g. [2, 45, 7, 23,...9]
		context_word = Input((1,), dtype='int32')

		# feed the words into the model using the Keras <Embedding> layer. This is the hidden layer 
		# from whose weights we will get the word embeddings.
		context_embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, name='context_embed_layer',
									embeddings_initializer='glorot_uniform',
									input_length=1)(context_word)

		# at this point, the input would of the shape (num_inputs x 1 x embed_size) and has to be flattened 
		# or reshaped into a (num_inputs x embed_size) tensor.
		context_input = Reshape((EMBED_SIZE, ))(context_embedding)
		merged_inputs = Dot(axes=-1, normalize=False)([target_input, context_input])
		label = Dense(1, 'sigmoid')(merged_inputs)

		# label is the output of step D.
		model = Model(inputs=[target_word, context_word], outputs=[label])

		model.compile(loss='mean_squared_error', optimizer='adam')

		h.write_report(str(model.summary()))

		NUM_EPOCHS = 3

		for epoch in range(0, NUM_EPOCHS):

			epoch_loss = 0

			for i, sent_examples in enumerate(skip_grams):

				target_wds = np.array([pair[0] for pair in sent_examples[0]], dtype='int32')
				context_wds = np.array([pair[1] for pair in sent_examples[0]], dtype='int32')
				labels = np.array(sent_examples[1], dtype='int32')
				X = [target_wds, context_wds]
				Y = labels

				if i % 5000 == 0: 

					print('Processed %d sentences' %i)
		        
				epoch_loss += model.train_on_batch(X, Y)

			print('Processed all %d sentences' %i)
			print('Epoch:', epoch, 'Loss:', epoch_loss, '\n')
		
		model.save("../output/skip.model")

		word_embeddings = model.get_layer('target_embed_layer').get_weights()[0] 

		# should return (VOCAB_SIZE, EMBED_SIZE)
		print(word_embeddings.shape)
		print(DataFrame(word_embeddings, index=idx2word.values()).head(10))

		similarity_matrix = cosine_similarity(word_embeddings)

		# should print(VOCAB_SIZE, VOCAB_SIZE)
		print(similarity_matrix.shape)

		search_terms = ['death', 'life', 'good', 'bad', 'man', 'woman', 'happy', 'unhappy', 'obama', 'trump', 'book', 'school', 'sex', 'apple', 'movie', 'university', 'london', 'russia', 'army', 'feminism', 'girl', 'boy']
		similar_words = dict()
		df = DataFrame(similarity_matrix, idx2word.values(), idx2word.values())

		for term in search_terms:

			similar_words[term] = df[term].abs().sort_values(ascending=False).iloc[1:11].index.to_list()

		print(similar_words)
		h.write_report(similar_words)
		h.plot_tsne(word_embeddings, idx2word)

		return model