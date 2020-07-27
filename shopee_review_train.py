import flair
from flair.datasets import ClassificationCorpus, TREC_6
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus, Sentence
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, BertEmbeddings,XLNetEmbeddings, RoBERTaEmbeddings, TransformerXLEmbeddings, DocumentRNNEmbeddings, CharacterEmbeddings, TransformerDocumentEmbeddings
from typing import List
from flair.datasets import ColumnCorpus
from pathlib import Path
import os
import sys
import time
import torch
from torch.optim.adam import Adam
import argparse

def train(
    review_category,
    params,
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
    print('loading training corpus from %s'%(params.data_folder))
    corpus: Corpus = ClassificationCorpus(params.data_folder,
                train_file= review_category+'_train.txt',
                test_file= review_category+'_test.txt',
                dev_file= review_category+'_dev.txt')
    label_dict = corpus.make_label_dictionary()
    print('labels: ',label_dict)
    if eval(params.transformer):
        print('initializing transformer document embeddings using %s ...'%(params.transformer_pretrain_lm))
        # 3. initialize transformer document embeddings (many models are available)
        document_embeddings = TransformerDocumentEmbeddings(params.transformer_pretrain_lm, fine_tune=True)
    else:
        print('initializing document embeddings')
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
            #XLNetEmbeddings()]
        # Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
        document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(word_embeddings,
                                                    hidden_size=512,
                                                    reproject_words=True,
                                                    reproject_words_dimension=256,
                                                    )
    if not update_model:
        print('building review_analysis classifier ...')
        # create the text classifier
        classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)
        # initialize the text classifier trainer
        print("initializing review_analysis classifier's trainer")
        trainer = ModelTrainer(classifier, corpus, optimizer=Adam)
    else:
        # continue trainer at later point
        checkpoint_path = params.checkpoint_dir+'/%s/checkpoint.pt'%(review_category)
        print('loading checkpoint from %s'%(checkpoint_path))
        trainer = ModelTrainer.load_checkpoint(checkpoint_path, corpus)
    ####### training the model
    print("training the review_category: %s model ..."%(review_category))
    try:
        trainer.train(params.checkpoint_dir+'/%s'%(review_category),
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
    except:
        print('chuncking batch ... by %d'%(params.mini_batch_chunk_size))
        trainer.train(params.checkpoint_dir+'/%s'%(review_category),
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
        max_epochs=max_epochs,
        mini_batch_chunk_size=params.mini_batch_chunk_size)

def get_parser():
    parser = argparse.ArgumentParser(description="review classification")
    # main parameter
    parser.add_argument("--which_gpu", type=int, default=-1, help="-1 means don't train on gpu; number is larger than -1 means which gpu card to train")
    parser.add_argument("--category",type=int, default="1")
    parser.add_argument("--data_folder",type=str, default="/raid/yihui/review_analysis")
    parser.add_argument("--checkpoint_dir",type=str, default="/raid/yihui/review_analysis/taggers")
    # model parameter
    parser.add_argument("--update_model",type=str, default="False", help="whether continue to train based on a previous checkpoint")
    parser.add_argument("--learning_rate",type=float, default=0.001)
    parser.add_argument("--embeddings_storage_mode", type=str, default='gpu')
    parser.add_argument("--checkpoint", type=str, default= 'True', help="whether record checkpoint or not but final model's checkpoint will be recorded by default. However, if there's an accidental stop and you don't record checkpoint, then there will not save anything. ")
    parser.add_argument("--batch_growth_annealing", type=str, default= 'True')
    parser.add_argument("--weight_decay", type=float, default= 1e-4)
    parser.add_argument("--shuffle", type=str, default='True')
    parser.add_argument("--train_with_dev", type=str, default='True')
    parser.add_argument("--mini_batch_size", type=int, default=2)
    parser.add_argument("--mini_batch_chunk_size", type=int, default=32, help='optionally set this if transformer is too much for your machine')
    parser.add_argument("--maxi_batch_size",type=int, default=128)
    parser.add_argument("--anneal_factor",type=float, default=0.5,help='lr decreases according to patience number.')
    parser.add_argument("--patience", type=int, default=2,help="the number of times that loss doesn't improve will trigger lr decrease by anneal_factor."
    parser.add_argument("--max_epochs", type=int, default=150)
    # which model
    parser.add_argument("--transformer", type=str, default='False')
    parser.add_argument("--transformer_pretrain_lm", type=str, default='roberta-large' ,help='See https://huggingface.co/transformers/pretrained_models.html for more options.')
    return parser


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    # project setting
    categories = [i for i in range(1,6)]
    print('categories in this project: %s'%(str(categories)))
    category_index = params.category
    review_category = categories[category_index]
    print('category type: %s'%(review_category))
    # setting available cuda
    if torch.cuda.is_available():
        cuda_core_num = params.which_gpu
        flair.device = torch.device('cuda:%d'%(cuda_core_num))
        print("setting cuda number to be %d"%(cuda_core_num))
    else:
        print("no cuda is available now")

    train(
        review_category = review_category,
        update_model= eval(params.update_model),
        learning_rate=params.learning_rate,
        embeddings_storage_mode=params.embeddings_storage_mode,
        checkpoint= eval(params.checkpoint),
        batch_growth_annealing= eval(params.batch_growth_annealing),
        weight_decay = params.weight_decay,
        shuffle=params.shuffle,
        train_with_dev=eval(params.train_with_dev),
        mini_batch_size=params.mini_batch_size,
        maxi_batch_size=params.maxi_batch_size,
        anneal_factor=params.anneal_factor,
        patience=params.patience,
        max_epochs=params.max_epochs,
        params = params
    )
