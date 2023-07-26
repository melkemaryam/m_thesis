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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from helper import Helper
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime

class Skip():

	def create_train(self, data):

		re = Read_data()
		h = Helper()

		#data = data['tokenised']

		word2idx = re.prepare_data(data)
		idx2word = re.get_idx2word(word2idx)

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

		model.compile(loss='mean_squared_error', optimizer='rmsprop')

		model.summary()
		h.write_report(model.summary())

		NUM_EPOCHS = 5

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

		
		word_embeddings = model.get_layer('target_embed_layer').get_weights()[0] 

		# should return (VOCAB_SIZE, EMBED_SIZE)

		print(word_embeddings.shape)

		print(DataFrame(word_embeddings, index=idx2word.values()).head(10))

		similarity_matrix = cosine_similarity(word_embeddings)

		# should print(VOCAB_SIZE, VOCAB_SIZE)
		print(similarity_matrix.shape)

		search_terms = ['death', 'life', 'good', 'bad', 'man', 'woman']

		similar_words = dict()
		df = DataFrame(similarity_matrix, idx2word.values(), idx2word.values())

		for term in search_terms:

			similar_words[term] = df[term].abs().sort_values(ascending=False).iloc[1:6].index.to_list()

		print(similar_words)
		h.write_report(similar_words)

		tsne = TSNE(perplexity=3, n_components=2, init='pca', n_iter=5000, method='exact')
		np.set_printoptions(suppress=True)
		plot_only = 50 

		T = tsne.fit_transform(word_embeddings[:plot_only, :])
		labels = [idx2word[i+1] for i in range(plot_only)]
		plt.figure(figsize=(14, 8))
		plt.scatter(T[:, 0], T[:, 1])
		for label, x, y in zip(labels, T[:, 0], T[:, 1]):
			plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points', ha='right', va='bottom')                      	                        

		plt.savefig("../plots/tsne_" + datetime.now().strftime("%Y%m%d-%H%M"))
		h.write_report("![](../plots/tsne_" + datetime.now().strftime("%Y%m%d-%H%M")+ ".png)")
		
		plt.show()

		return model