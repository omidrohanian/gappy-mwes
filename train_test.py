import os, pickle, subprocess
import numpy as np
from keras.callbacks import ModelCheckpoint
from evaluation import labels2Parsemetsv

from sklearn.model_selection import KFold
from models.tag_models import Tagger 

class Train_Test():
	"""This class contains methods to train, test or cross-validate models. 

	Attributes:
		* pos: Boolean value specifying whether POS information is used 
		* tagger_name: name of the model that is to be trained 
		* tagger: a compiled instance of the model (with the name tagger_name) 
		* data: an instance of the classs Data built at the preprocessing step 
	"""
	def __init__(self, pos, tagger_name, tagger, data):
		self.pos = pos 
		self.tagger_name = tagger_name
		self.tagger = tagger 
		self.data = data
		self.w2v = self.data.word2vec_dir

		# preparing the name for the folder results corresponding to the name of the model, settings and language.
		self.res_dir = "./results/" + self.data.testORdev + "_{}".format(self.data.lang)+self.tagger_name+"_results"
		if not os.path.exists(self.res_dir):
			os.makedirs(self.res_dir)

	def train(self, epoch, batch_size):
		
		filepath = self.res_dir + "/weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max', period=10, save_weights_only=True)
		callbacks_list = [checkpoint]

		# preparing the inputs
		inputs = []
		if "elmo" in self.tagger_name.lower():
			inputs = [self.data.train_weights]
		if self.w2v:
			inputs += [self.data.X_train_enc]
		if self.pos:
			inputs += [self.data.pos_train_enc]
		if self.data.depAdjacency_gcn:
			inputs += self.data.train_adjacency_matrices
		print("len(inputs)", len(inputs))
		if len(inputs) == 1:
			self.tagger.fit(inputs[0],           
			 				self.data.y_train_enc, 
			 				validation_split=0, 
			 				batch_size=batch_size, 
			 				epochs=epoch, 
			 				callbacks=callbacks_list)

		else:
			self.tagger.fit(inputs, 
						   self.data.y_train_enc, 
						   epochs=epoch,
						   validation_split=0, 
						   batch_size=batch_size, 
						   callbacks=callbacks_list)
		
	def test(self, data_path):
		inputs = []
		if "elmo" in self.tagger_name.lower():
			inputs = [self.data.test_weights]
		if self.w2v:
			inputs += [self.data.X_test_enc]
		if self.pos:
			inputs += [self.data.pos_test_enc]
		if self.data.depAdjacency_gcn:
			inputs += self.data.test_adjacency_matrices
		
		if len(inputs)==1:
			preds = self.tagger.predict(inputs[0], batch_size=16, verbose=1)
		else:
			preds = self.tagger.predict(inputs, batch_size=16, verbose=1)

		final_preds = []
		for i in range(len(self.data.X_test_enc)):
			pred = np.argmax(preds[i],-1)
			pred = [self.data.idx2l[p] for p in pred]
			final_preds.append(pred)
		# preparing the name for the prediction file corresponding to the name of the model, settings and language.
		prediction_file_name = self.res_dir + '/predicted_{}'.format(self.data.lang)+'_'+self.tagger_name
		# save the predicted labels to a pickle list
		with open(prediction_file_name+'.pkl', 'wb') as f:
		    pickle.dump(final_preds, f)
		with open(prediction_file_name+'.pkl', 'rb') as f:
		    labels1 = pickle.load(f)
		if self.data.testORdev == "TEST":	# we have DEV as part of training and are evaluating the test
			labels2Parsemetsv(labels1, data_path+'{}/test.cupt'.format(self.data.lang), prediction_file_name+'_system.cupt')

			with open(self.res_dir + '/eval'.format(self.data.lang)+self.tagger_name+'.txt', 'w') as f:
				f.write(subprocess.check_output([data_path+"bin/evaluate_v1.py", "--train", data_path+"{}/train.cupt".format(self.data.lang), "--gold", data_path+"{}/test.cupt".format(self.data.lang), "--pred", prediction_file_name+"_system.cupt" ]).decode())
		else:
			labels2Parsemetsv(labels1, data_path+'/{}/dev.cupt'.format(self.data.lang), prediction_file_name+'_system.cupt')

			with open(self.res_dir + '/eval'.format(self.data.lang)+self.tagger_name+'.txt', 'w') as f:
				f.write(subprocess.check_output([data_path+"bin/evaluate_v1.py", "--train", data_path+"{}/train.cupt".format(self.data.lang), "--gold", data_path+"{}/dev.cupt".format(self.data.lang), "--pred", prediction_file_name+"_system.cupt" ]).decode())

	def cross_validation(self, epoch, batch_size, data_path):
		if self.data.testORdev == "CROSS_VAL": 
			self.res_dir="./results/CROSSVAL_{}".format(self.data.lang)+"_"+self.tagger_name+"_results"
		else:
			pass
		if not os.path.exists(self.res_dir):
			os.makedirs(self.res_dir)

		kf = KFold(n_splits=5)
		i=0
		final_preds = [0]*len(self.data.X_train_enc)
		for train_index, test_index in kf.split(self.data.X_train_enc):
			print("Running Fold", i+1, "/", "5")
			X_train, X_test = self.data.X_train_enc[train_index], self.data.X_train_enc[test_index]
			pos_train, pos_test = self.data.pos_train_enc[train_index], self.data.pos_train_enc[test_index]
			y_train, y_test = self.data.y_train_enc[train_index], self.data.y_train_enc[test_index]
			inputs = []
			if "elmo" in self.tagger_name.lower():
				X_train, X_test = self.data.train_weights[train_index], self.data.train_weights[test_index]
				inputs += [X_train]
			if self.pos:
				inputs += [pos_train]
			X_train_adj, X_test_adj = [], []
			if self.data.depAdjacency_gcn:
					for j in range(len(self.data.train_adjacency_matrices)):
						X_train_adj.append(self.data.train_adjacency_matrices[j][train_index])
						X_test_adj += [self.data.train_adjacency_matrices[j][test_index]]
					inputs += X_train_adj
			print(X_train.shape)

			model = None # Clearing the NN.
			model = Tagger(self.data, self.data.max_length, self.data.input_dim, self.data.n_poses, self.data.n_classes, "")
			model = getattr(model, self.tagger_name)() 
			#if "elmo" in self.tagger_name.lower():
			#	model.fit(train_text, y_train, validation_split=0, batch_size=10, epochs=1)

			if len(inputs)==1:
				model.fit(X_train, 
                                y_train, 
                                validation_split=0, 
                                batch_size=batch_size, 
                                epochs=epoch)
			else:
				model.fit(inputs, 
                                y_train, 
                                validation_split=0, 
                                batch_size=batch_size, 
                                epochs=epoch)
			i+=1

			for t in test_index:
				inputs = [np.array([self.data.train_weights[t]])]
				if self.pos:
					inputs += [np.array([self.data.pos_train_enc[t]])]
				if self.data.depAdjacency_gcn:
					inputs += [np.array([self.data.train_adjacency_matrices[j][t]]) for j in range(len(self.data.train_adjacency_matrices))]

				if len(inputs)==1:
					pred = model.predict(np.array([self.data.train_weights[t]])) #.reshape(1, -1))
				else:
					pred = model.predict(inputs)
				pred = np.argmax(pred,-1)[0]
				pred = [self.data.idx2l[p] for p in pred]
				final_preds[t] = pred

		prediction_file_name = self.res_dir + '/predicted_{}'.format(self.data.lang)+'_'+self.tagger_name
		with open(prediction_file_name+'.pkl', 'wb') as f:
		    pickle.dump(final_preds, f)
		with open(prediction_file_name+'.pkl', 'rb') as f:
		    labels1 = pickle.load(f)
		print("len(labels1)",len(labels1))
		labels2Parsemetsv(labels1, data_path+'{}/train.cupt'.format(self.data.lang), prediction_file_name+'_system.cupt')

		with open(self.res_dir + '/eval'.format(self.data.lang)+self.tagger_name+'.txt', 'w') as f:
			f.write(subprocess.check_output([data_path+"bin/evaluate_v1.py", "--gold", data_path+"{}/train.cupt".format(self.data.lang), "--pred", prediction_file_name+"_system.cupt" ]).decode())
	
