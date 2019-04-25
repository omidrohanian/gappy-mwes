# Overview
This repository contains the code and sample data for the experiments related to our NAACL 2019 paper "Bridging the Gap: Attending to Discontinuity in Identification of Multiword Expressions".
 
## Prerequisites
```
pip install -r ./requirements.txt
```

## Run a sample 

To check if you have the proper dependencies and see a sample output, simply run `main.py`. The results (including the predicted labels and the evaluation performance) will be saved in the directory results/{model settings}.

## Run your own tagger
Please follow the steps below to run your own model:

##### 1 - Copy your train, test and dev .cupt files under the directory data/{language}/.

##### 2 - Download pre-trained ELMo embeddings.

The models are trained using pre-trained ELMo embeddings. To obtain representations for your inputs, follow the instructions at [ELMo For Many Langs](https://github.com/HIT-SCIR/ELMoForManyLangs). Save the output h5py file (as ELMo_EN, ELMo_FR, etc) and place the files in the `./embeddings` folder. 

##### 3 - Set all variables (language, tagging model etc) in lines 10-22 of `main.py` .

##### 4 - run `main.py`.

### Project Structure
In the interest of readability, we provide an overview of the structure of our project: 

```bash
.
├── README.md
├── corpus.py
├── corpus_reader.py
├── data
│   ├── EN_SAMPLE
│   │   ├── dev.cupt
│   │   ├── test.cupt
│   │   └── train.cupt
│   └── bin
│       ├── evaluate_v1.py
│       └── tsvlib.py
├── embeddings
│   └── ELMo_EN_SAMPLE
├── evaluation.py
├── main.py
├── models
│   ├── layers.py
│   └── tag_models.py
├── preprocessing.py
├── requirements.txt
├── results
└── train_test.py
```
- `corpus.py`, and `corpus_reader.py`: read the data in the conll format and contain methods that is used by the preprocessor. 
- `embeddings`: pre-trained embedding files with the format ELMO_{EN|FR|FA|DE}. It contains a sample file (ELMo_EN_SAMPLE) for the trial run.  
- `evaluation.py`: contains the script for evaluation. 
- `preprocessing.py`: prepares data in the proper format and loads ELMo embeddings. 
- `layers.py`: self-attention, GCN, and highway layers are defined here. 
- `tag_models.py`: our models are all defined here. This part depends on `layers.py`.
- `train_test.py`: contains the functions for training and testing the models. 
- `main.py`: the main part connesting all the other scripts. You can specify the main variables here (language, epochs, model, ...).
- `data`: contains sample data (in `EN_SAMPLE`) including 5, 2 and 4 sentences in train, dev and test respectively for a trial run. To obtain the data used in the experiments, download train, test, and dev files for each language from [Parseme's gitlab page](https://gitlab.com/parseme/sharedtask-data/tree/master/1.1). 
- `bin`: contains the evaluation code of the PARSEME shared task on identifying VMWEs.
- `requirements.txt`: contains the names and versions of the required dependencies. 
- `results`: results appear here after `main.py` is run. 

### TL;DR

If you only care about the model architecture, have a look at `./models/tag_models`. Self-attention and GCN are defined in the separate file `./models/layers.py`.

### Reference
```
@article{Rohanian2019,
  author    = {Omid Rohanian and
               Shiva Taslimipoor and
               Samaneh Kouchaki and
               Le An Ha and
               Ruslan Mitkov},
  title     = {Bridging the Gap: Attending to Discontinuity in Identification of
               Multiword Expressions},
  year      = {2019},
  url       = {http://arxiv.org/abs/1902.10667},
  archivePrefix = {arXiv},
  eprint    = {1902.10667},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1902-10667},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
