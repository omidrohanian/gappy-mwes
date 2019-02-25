import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np 
from preprocessing import Data 
from train_test import Train_Test 
from models.tag_models import Tagger 
from evaluation import labels2Parsemetsv

DEVorTEST = "TEST"		    	# 'TEST' or 'DEV' or 'CROSS_VAL'
LANG = "EN_SAMPLE"		    	# specify the name of the language (i.e. EN|FR|FA|DE)
MODEL = "model_ELMo_spect_gcn"		# specify the name of the training model to be used (models are defined in rnn_models.py)
POS = False				# set True to use POS information (This is set to False in the reported results as the improvement was not consistent across all the models)
DEP_ADJACENCY_GCN = True    		# True for models that require adjacency, and False otherwise
EPOCHS = 50				# the number of epochs for trainig the model 
BATCH_SIZE = 100			# set training batch size 
DATAPATH = "./data/"			# for each language, place train, dev, and test files in an appropriately named folder (i.e. EN|FR|FA|DE)		
initial_weight=''			# training can be resumed from a checkpoint if a saved file of weights is assigned to this variable 
POSITION_EMBED = False			# position embedding wasn't used (as explained in Sec 2.2 of the paper)  
WV_DIR = "" 				# the directory for word2vec embeddings (if applicable, pass an empty string) 
ELMO_PATH = "./embeddings"	    	# Place your embedding files in this directory (with the format ELMO_{EN|FR|FA|DE})

def run_model():
	d = Data(LANG, DEVorTEST, WV_DIR, ELMO_PATH, MODEL, DEP_ADJACENCY_GCN, POSITION_EMBED)
	d.load_data(DATAPATH)	 	# This loads train, dev, and test if available, and also word2vec and ELMo where relevant 

	model = Tagger(d, d.max_length, d.input_dim, d.n_poses, d.n_classes, initial_weight)  
	tagger = getattr(model, MODEL)() # choose the specified tagging model

	T = Train_Test(POS, MODEL, tagger, d)
	if DEVorTEST == "CROSS_VAL":
                T.cross_validation(EPOCHS, BATCH_SIZE, DATAPATH)
	else:                
                T.train(EPOCHS, BATCH_SIZE)
                T.test(DATAPATH)	# We pass DATAPATH to this function to be used for evaluation

run_model()




