import pickle, os
import numpy as np
from collections import Counter 
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, \
						concatenate, Conv1D, BatchNormalization, CuDNNLSTM, Lambda, \
						Multiply, Add, Activation, Flatten, MaxPooling1D 
from keras.layers.wrappers import TimeDistributed
from keras import regularizers
from keras.regularizers import l2
import keras.initializers
from models.layers import * 

class Tagger(object):
	def __init__(self, data, max_length, input_dim, n_poses, n_classes, initial_weight=''):
		self.max_length = data.max_length
		self.input_dim = input_dim
		self.n_poses = n_poses
		self.n_classes = n_classes
		self.initial_weight = initial_weight
		self.data = data

	    ################################################
	    ############# The baseline models ##############		

	def model_ELMo(self):
		"""the baseline model. similar to SHOMA but uses ELMo embeddings in lieu of word2vec"""
		embed = Input(shape=(self.max_length,self.input_dim))
		conv1 = Conv1D(200, 2, activation="relu", padding="same", name='conv1')(embed)
		conv2 = Conv1D(200, 3, activation="relu", padding="same", name='conv2')(embed)
		conc = concatenate([conv1, conv2])
		if self.max_length < 300:
			lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(conc)
		else:
			lstm = Bidirectional(CuDNNLSTM(300,return_sequences=True, name='lstm'))(conc)
			lstm = Dropout(0.5)(lstm)
		output = TimeDistributed(Dense(self.n_classes, activation='softmax', name='dense'), name='time_dist')(lstm)		
		model = Model(inputs=embed, outputs=output)
		if self.initial_weight:
			model.load_weights(self.initial_weight)
		model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['mae', 'acc'])
		print(model.summary())
		return model	

	def model_ELMo_withPOS(self):	
		"""the baseline model with addition of POS features"""
		elmo_embed = Input(shape=(self.max_length,self.input_dim))
		posInput = Input(shape=(self.max_length,self.n_poses,))
		embed = concatenate([elmo_embed, posInput])
		conv1 = Conv1D(200, 2, activation="relu", padding="same", name='conv1')(embed)
		conv2 = Conv1D(200, 3, activation="relu", padding="same", name='conv2')(embed)
		conc = concatenate([conv1, conv2])
		if self.max_length < 300:
			lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(conc)
		else:
			lstm = Bidirectional(CuDNNLSTM(300,return_sequences=True, name='lstm'))(conc)
			lstm = Dropout(0.5)(lstm)
		output = TimeDistributed(Dense(self.n_classes, activation='softmax', name='dense'), name='time_dist')(lstm)	
		#output = Dense(n_classes, activation='softmax', name='dense')(embed)	
		model = Model(inputs=[elmo_embed, posInput], outputs=output)
		if self.initial_weight:
			model.load_weights(self.initial_weight)
		model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['mae', 'acc'])
		print(model.summary())
		return model

	def model_ELMo_withW2V(self):	
		"""the baseline model. uses a concatenation of word2vec and ELMo embeddings"""
		elmo_embed = Input(shape=(self.max_length,self.input_dim))
		visible = Input(shape=(self.max_length,))
		embed = self.data.embedding_layer(visible)
		embed = concatenate([elmo_embed, embed])
		conv1 = Conv1D(200, 2, activation="relu", padding="same", name='conv1')(embed)
		conv2 = Conv1D(200, 3, activation="relu", padding="same", name='conv2')(embed)
		conc = concatenate([conv1, conv2])
		lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(conc)
		output = TimeDistributed(Dense(self.n_classes, activation='softmax', name='dense'), name='time_dist')(lstm)  
		model = Model(inputs=[elmo_embed, visible], outputs=output)
		if self.initial_weight:
		    model.load_weights(self.initial_weight)
		model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['mae', 'acc'])
		print(model.summary())
		return model


	def model_withPOS(self): 
		"""baseline model SHOMA from the shared task. Uses word2vec"""
		visible = Input(shape=(self.max_length,))
		embed = self.embedding_layer(visible)
		# posInput = Input(shape=(max_length, 17))
		posInput = Input(shape=(self.max_length,self.n_poses,))
		embed = concatenate([embed, posInput])
		conv1 = Conv1D(200, 2, activation="relu", padding="same", name='conv1')(embed)
		conv2 = Conv1D(200, 3, activation="relu", padding="same", name='conv2')(embed)
		conc = concatenate([conv1, conv2])
		lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(conc)
		output = TimeDistributed(Dense(self.n_classes, activation='softmax', name='dense'), name='time_dist')(lstm)  
		model = Model(inputs=[visible, posInput], outputs=output)
		if self.initial_weight:
		    model.load_weights(self.initial_weight)
		model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['mae', 'acc'])
		print(model.summary())
		return model	

	    ################################################
	    ############# The proposed models ##############

	def model_ELMo_Att_Based(self):
		"""the model referred to as Att-based in the paper"""

		# some variables to set 
		hidden = 300
		dim = hidden * 2 
		heads = 4

		embed = Input(shape=(self.max_length,self.input_dim))
		# stacked CNNs
		c1 = Conv1D(100, 1, activation="relu", padding="same", dilation_rate=1)(embed)
		c2 = Conv1D(100, 1, activation="relu", padding="same", dilation_rate=1)(c1) 
		 # parallel CNN
		cnn = Conv1D(100, 1, activation="relu", padding="same", dilation_rate=1)(embed)

		input_ = BatchNormalization()(concatenate([cnn, c2])) 

		lstm1 = Bidirectional(LSTM(hidden, return_sequences=True, name='lstm1', dropout=0.5, recurrent_dropout=0.2))(input_)
		att = MultiHeadAttention(n_head=heads, d_model=dim, d_k=dim, d_v=dim, dropout=0.5, mode=1)
		self_att = att(q=lstm1, k=lstm1, v=lstm1)[0]

		output = TimeDistributed(Dense(self.n_classes, activation='softmax', name='dense'), name='time_dist')(self_att)	
		model = Model(inputs=embed, outputs=output)
		if self.initial_weight:
			model.load_weights(self.initial_weight)
		model.compile(
						optimizer='adam',
						loss='categorical_crossentropy',
						metrics=['mae','acc'])
		print(model.summary())
		return model 


	def model_ELMo_spect_gcn(self):		
		"""the model based on graph convolutional network. gcn is followed by bi-lstm"""
		input_X = Input(shape=(self.max_length,self.input_dim), name='x-data')
		input_A = [Input(shape=(self.max_length, self.max_length), name='A-edge_{}'.format(i)) for i in range(self.data.depAdjacency_gcn)]	# dtype='int32'

		H = SpectralGraphConvolution(256, activation='relu', self_links=True, 
		         backward_links=True)([input_X] + input_A)
		H = Dropout(0.5)(H)
		if self.max_length < 300:
			H = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(H)
		else:
			print('CuDNNLSTM')
			H = Bidirectional(CuDNNLSTM(300,return_sequences=True, name='lstm'))(H)
			H = Dropout(0.5)(H)
		output = Dense(self.n_classes, activation='softmax')(H)

		model = Model(inputs=[input_X] + input_A, outputs=output)
		if self.initial_weight:
			model.load_weights(self.initial_weight)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae','acc'])
		print(model.summary())
		return model


	def model_ELMo_H_combined(self):
		"""concat gcn and self_att and then apply highway on the output, finally concatenate with gcn. num of highway layers = 3, bias = -2"""

		# some variables to set 
		hidden = 300
		dim = 200 #hidden * 2 
		k = 1
		n_attn_heads = 4
		l2_reg = 5e-4/2
		dropout_rate = 0.5
		n_layers = 4 

		X_in = Input(shape=(self.max_length,self.input_dim), name='x-data')
		A_in = [Input(shape=(self.max_length, self.max_length), name='A-edge_{}'.format(i)) for i in range(self.data.depAdjacency_gcn)]

		# stacked CNNs
		cn_stacked = Conv1D(100, k, activation="relu", padding="same")(X_in)
		cn_stacked = Conv1D(100, k, activation="relu", padding="same")(cn_stacked) 
		# parallel CNN
		cn = Conv1D(100, k, activation="relu", padding="same")(X_in) 
		cn_concat = concatenate([cn, cn_stacked])         
		cn_concat = Dropout(dropout_rate)(cn_concat)

		att = MultiHeadAttention(n_head=n_attn_heads, d_model=dim, d_k=dim, d_v=dim, dropout=0.5, mode=1)
		self_att = att(q=cn_concat, k=cn_concat, v=cn_concat)[0]

		gcn = SpectralGraphConvolution(200, activation='relu')([X_in] + A_in)
		gcn = Dropout(dropout_rate)(gcn)


		conc = concatenate([gcn, self_att])
		gate = BatchNormalization()(Highway(n_layers = n_layers, value=conc, gate_bias=-2)) 
		if self.max_length < 300:
			lstm = Bidirectional(LSTM(hidden,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(gate)
		else:
			print('CuDNNLSTM')
			lstm = Bidirectional(CuDNNLSTM(hidden,return_sequences=True, name='lstm'))(gate)
			lstm = Dropout(0.5)(lstm)

		output = Dense(self.n_classes, activation='softmax', name='dense')(lstm)

		# Build the model
		model = Model(inputs=[X_in] + A_in, outputs=output)
		if self.initial_weight:
			model.load_weights(self.initial_weight)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae','acc'])
		print(model.summary())
		return model

