import flair
from flair.datasets import ClassificationCorpus, TREC_6
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus, Sentence
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, BertEmbeddings,XLNetEmbeddings, RoBERTaEmbeddings, TransformerXLEmbeddings, DocumentRNNEmbeddings, CharacterEmbeddings
from typing import List
from flair.datasets import ColumnCorpus
from pathlib import Path
import os
import sys
import time
import torch
from torch.optim.adam import Adam

categories = [i for i in range(1,6)]
data_folder = '/raid/yihui/review_analysis'
word_embeddings= [
        WordEmbeddings('glove'),
        # comment in this line to use character embeddings
        CharacterEmbeddings(),
        # comment in these lines to use flair embeddings
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
        BertEmbeddings(),
        # TransformerXLEmbeddings(),
        #RoBERTaEmbeddings(),
        #XLNetEmbeddings()
    ]
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(word_embeddings,
                                                                    hidden_size=512,
                                                                    reproject_words=True,
                                                                    reproject_words_dimension=256,
                                                                    )

def train(
    review_category,
    update_model= False,
    learning_rate=0.01,
    embeddings_storage_mode='gpu',
    checkpoint= True,
    batch_growth_annealing= True,
    weight_decay = 1e-4,
    shuffle=True,
    train_with_dev=True,
    mini_batch_size=2,
    maxi_batch_size=128,
    anneal_factor=0.5,
    patience=2,
    max_epochs=150
    ):
    review_category = str(review_category)
    corpus: Corpus = ClassificationCorpus('/raid/yihui/review_analysis',
                train_file= review_category+'_train.txt',
                test_file= review_category+'_test.txt',
                dev_file= review_category+'_dev.txt')
    label_dict = corpus.make_label_dictionary()
    print('labels: ',label_dict)
    if not update_model:
        print('building review_analysis classifier ...')
        # create the text classifier
        classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)
        # initialize the text classifier trainer
        trainer = ModelTrainer(classifier, corpus, optimizer=Adam)
        # start the training
        print('starting to train %s model'%(review_category))
    else:
        # continue trainer at later point
        checkpoint_path = data_folder+'/resources/taggers/%s/checkpoint.pt'%(review_category)
        print('loading checkpoint from %s'%(checkpoint_path))
        trainer = ModelTrainer.load_checkpoint(checkpoint_path, corpus)
        print("training the model ...")
    ####### training the model
    trainer.train(data_folder+'resources/taggers/%s'%(review_category),
    learning_rate=learning_rate,
    embeddings_storage_mode=embeddings_storage_mode,
    checkpoint= checkpoint,
    batch_growth_annealing= batch_growth_annealing,
    weight_decay = weight_decay,
    shuffle=shuffle,
    train_with_dev=train_with_dev,
    mini_batch_size=mini_batch_size,
    maxi_batch_size=maxi_batch_size,
    anneal_factor=anneal_factor,
    patience=patience,
    max_epochs=max_epochs)

if __name__ == '__main__':
    # for review_category in categories:
    category_index = int(sys.argv[1])
    review_category = categories[category_index]
    print('category type: %s'%(review_category))
    # setting available cuda
    if torch.cuda.is_available():
        cuda_core_num = int(sys.argv[2])
        flair.device = torch.device('cuda:%d'%(cuda_core_num))
        print("setting cuda number to be %d"%(cuda_core_num))
    else:
        print("no cuda is available now")
    train(
    review_category = review_category,
    update_model= False,
    learning_rate=0.0002,
    embeddings_storage_mode='gpu',
    checkpoint= True,
    batch_growth_annealing= True,
    weight_decay = 1e-4,
    shuffle=True,
    train_with_dev=True,
    mini_batch_size=64,
    maxi_batch_size=128,
    anneal_factor=0.5,
    patience=2,
    max_epochs=150
    )
