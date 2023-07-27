import keras
import numpy as np
from keras.layers import Lambda, GlobalAveragePooling1D, Dense, Embedding
from keras import backend as K
from keras.models import Sequential
import matplotlib.pyplot as plt

from keras.layers import LSTM, RNN, Dropout, Input, LeakyReLU, Bidirectional,Conv1D, GlobalMaxPooling1D
from keras.layers.core import Dense
from keras.models import Model

from transformers import DistilBertTokenizer, RobertaTokenizer 
import tqdm
from read_data import Read_data

distil_bert = 'distilbert-base-uncased' # Pick a pre-trained model

# Defining DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(distil_bert, do_lower_case=True, add_special_tokens=True,
                                                max_length=128, pad_to_max_length=True)


class Transformer():

	def tokenize(sentences, tokenizer, pad_length=128, pad_to_max_length=True):

		if type(sentences) == str:
			inputs = tokenizer.encode_plus(sentences, add_special_tokens=True, max_length=pad_length, pad_to_max_length=pad_to_max_length, return_attention_mask=True, return_token_type_ids=True)
			return np.asarray(inputs['input_ids'], dtype='int32'), np.asarray(inputs['attention_mask'], dtype='int32'), np.asarray(inputs['token_type_ids'], dtype='int32')

		input_ids, input_masks, input_segments = [],[],[]

		for sentence in sentences:
			inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=pad_length, pad_to_max_length=pad_to_max_length, return_attention_mask=True, return_token_type_ids=True)
			input_ids.append(inputs['input_ids'])
			input_masks.append(inputs['attention_mask'])
			input_segments.append(inputs['token_type_ids'])        

		return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32')

	def get_bert_inputs(examples_list, targets):
		
		input_ids=list()
		attention_masks=list()

		bert_inp=tokenize(examples_list, tokenizer)
		input_ids = bert_inp[0]
		attention_masks = bert_inp[1]

		targets = np.array(targets)

		return input_ids, attention_masks, targets

	def build_trans(self, data):

		re = Read_data()

		X_train, X_test, y_train, y_test = re.train_test_data(data)
		



