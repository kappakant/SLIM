# coding: utf-8

__author__ = "Ciprian-Octavian Truică"
__copyright__ = "Copyright 2021, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "ciprian.truica@upb.ro"
__status__ = "Production"

# Modifications to hyperparameters compared to Truica's implementation.

import os
import sys
from keras.utils import to_categorical
# helpers
import time

# classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import pandas as pd
import sys
import os
import os.path

# split data set
import multiprocessing as mp

# import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Embedding, GRU, Dropout, LSTM, Bidirectional, SimpleRNN, Input, Concatenate
# from keras.layers import Attention, Concatenate, Input
#from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.python.keras.preprocessing.text import Tokenizer
#from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import math
from keras import Model
import tensorflow as tf
from keras.layers import Conv1D, Flatten, MaxPooling1D, Reshape
import random as rnd

from simpletransformers.language_representation import RepresentationModel
from sentence_transformers import SentenceTransformer


#TFIDF_FOUT = "./models/tfidf_features"
MAX_FEATURES = 5000

MIN_DF = 5
# MAX_DF = 1
MAX_FEATURES=5000
LOWER = True
USE_IDF = True
SMOOTH_IDF =True
ANALYZER = 'word'

accuracies= []
precisions_micro = []
precisions_macro = []
recalls_micro = []
recalls_macro = []
execution_time = []

def evaluate(y_test, y_pred, modelName='GRU', iters=0):
    y_pred_norm = []

    for elem in y_pred:
        line = [ 0 ] * len(elem)
        try:
            # if an error appears here
            # get a random class
            elem[np.isnan(elem)] = 0
            line[elem.tolist().index(max(elem.tolist()))] = 1
        except:
            print("Error for getting predicted class")
            print(elem.tolist())
            line[rnd.randint(0, len(elem)-1)] = 1
        y_pred_norm.append(line)

    y_p = np.argmax(np.array(y_pred_norm), 1)
    y_t = np.argmax(np.array(y_test), 1)
    accuracy = accuracy_score(y_t, y_p)
    accuracies.append(accuracy)
    precision_micro = precision_score(y_t, y_p, average='micro')
    precisions_micro.append(precision_micro)
    precision_macro = precision_score(y_t, y_p, average='macro')
    precisions_macro.append(precision_macro)
    recall_micro = recall_score(y_t, y_p, average='micro')
    recalls_micro.append(recall_micro)
    recall_macro = recall_score(y_t, y_p, average='macro')
    recalls_macro.append(recall_macro)
    print("Accuracy", accuracy)
    print("Precision Micro", precision_micro)
    print("Precision Macro", precision_macro)
    print("Recall Micro", recall_micro)
    print("Recall Macro", recall_macro)
    print("Report", classification_report(y_t, y_p))
    print(confusion_matrix(y_t, y_p))
    return y_p, y_t

def splitDataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, shuffle = True, stratify = y)
    return X_train, X_test, np.array(y_train), np.array(y_test)

def getBERTEncoding(X_train, X_test, use_cuda=False):
    model = RepresentationModel(model_type="bert", model_name="bert-large-uncased", use_cuda=use_cuda)
    X_train_BERT = model.encode_sentences(X_train, combine_strategy="mean")
    X_test_BERT = model.encode_sentences(X_test, combine_strategy="mean")
    X_train_BERT = X_train_BERT.reshape((X_train_BERT.shape[0], 1, X_train_BERT.shape[1]))
    X_test_BERT = X_test_BERT.reshape((X_test_BERT.shape[0], 1, X_test_BERT.shape[1]))
    return X_train_BERT, X_test_BERT

def getRoBERTaEncoding(X_train, X_test, use_cuda=False):
    model = RepresentationModel(model_type="roberta", model_name="roberta-base", use_cuda=use_cuda)
    X_train_RoBERTa = model.encode_sentences(X_train, combine_strategy="mean")
    X_test_RoBERTa = model.encode_sentences(X_test, combine_strategy="mean")
    X_train_RoBERTa = X_train_RoBERTa.reshape((X_train_RoBERTa.shape[0], 1, X_train_RoBERTa.shape[1]))
    X_test_RoBERTa = X_test_RoBERTa.reshape((X_test_RoBERTa.shape[0], 1, X_test_RoBERTa.shape[1]))
    return X_train_RoBERTa, X_test_RoBERTa

def getBARTEncoding(X_train, X_test, use_cuda=False):
    model = SentenceTransformer('facebook/bart-large')
    # model = RepresentationModel(model_type="bart", model_name="facebook/bart-large", use_cuda=use_cuda)
    # X_train_BART = model.encode_sentences(X_train, combine_strategy=None)
    X_train_BART = model.encode(X_train)
    # X_test_BART = model.encode_sentences(X_test, combine_strategy=None)
    X_test_BART = model.encode(X_test)
    print(X_train_BART)
    X_train_BART = X_train_BART.reshape((X_train_BART.shape[0], 1, X_train_BART.shape[1]))
    X_test_BART = X_test_BART.reshape((X_test_BART.shape[0], 1, X_test_BART.shape[1]))
    return X_train_BART, X_test_BART

def prepareYTrainTestNN(y_train, y_test, num_classes):
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    print(y_test)
    return y_train, y_test

if __name__ =="__main__":
    FIN = sys.argv[1]
    USE_CUDA = bool(int(sys.argv[2])) # 0 - False, 1 - True
    print(USE_CUDA)
    NUM_ITER = int(sys.argv[3])

    dataSet = pd.read_csv(FIN, encoding = "utf-8")
    labels = dataSet['label'].unique()
    num_classes = len(labels)
    id2label = {}
    idx = 0
    for label in labels:
        id2label[idx] = label
        dataSet.loc[dataSet['label'] == label, 'label'] = idx
        idx += 1

    for key in id2label:
        print(key, id2label[key])

    X = dataSet['merged_info'].astype(str).to_list()
    #X = X[ : 100] + X[-100:]
    y = dataSet['label'].astype(int).to_list()
    #y = y[ : 100] + y[-100:]

    epochs_n = 20
    filters = 64
    units = 128
    no_attributes = int(units * 2) # no units
    # no_attributes = units # no units
    kernel_size = int(no_attributes/2)

    # randomizing features
    for idx in range(0, NUM_ITER):
        start_time = time.time()
        # Sampling examples
        X_train, X_test, y_train, y_test = splitDataset(X, y)

        x_vec_train = [ [], [], [], [] ]
        x_vec_test = [ [], [], [], [] ]
        print(y_train.shape, y_test.shape)
        y_vec_train, y_vec_test = prepareYTrainTestNN(y_train, y_test, num_classes)
        print(y_vec_train.shape, y_vec_test.shape)

        print("Bart")
        x_vec_train[0], x_vec_test[0] = getBARTEncoding(X_train, X_test, USE_CUDA)
        print(x_vec_train[0].shape)
        print(x_vec_test[0].shape)

        # RoBERTa encodings
        print("RoBERTa")
        x_vec_train[1], x_vec_test[1] = getRoBERTaEncoding(X_train, X_test, USE_CUDA)
        print(x_vec_train[1].shape)
        print(x_vec_test[1].shape)

        input_bart = Input(shape=(x_vec_train[0].shape[1], x_vec_train[0].shape[2]), name = 'BART_Input')
        model_bart = Bidirectional(LSTM(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'BART_BiLSTM_1')(input_bart)
        model_bart = Bidirectional(LSTM(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'BART_BiLSTM_2')(input_bart)
        model_bart = Bidirectional(LSTM(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'BART_BiLSTM_3')(input_bart)
        model_bart = Reshape((no_attributes, 1), name = 'BART_Reshape_1')(model_bart) # reshape to number of units
        model_bart = Conv1D(filters = filters, kernel_size=kernel_size, name = 'BART_CNN_1')(model_bart)
        model_bart = MaxPooling1D(name='BART_MaxPolling_1')(model_bart)
        model_bart = Flatten(name='BART_Flatten_1')(model_bart)

        input_roberta = Input(shape=(x_vec_train[1].shape[1], x_vec_train[1].shape[2]), name = 'BERT_Input')
        model_roberta = Bidirectional(LSTM(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'RoBERTa_BiLSTM_1')(input_roberta)
        model_roberta = Bidirectional(LSTM(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'RoBERTa_BiLSTM_2')(input_roberta)
        model_roberta = Bidirectional(LSTM(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'RoBERTa_BiLSTM_3')(input_roberta)
        model_roberta = Reshape((no_attributes, 1), name = 'RoBERTa_Reshape_1')(model_roberta) # reshape to number of units
        model_roberta = Conv1D(filters = filters, kernel_size=kernel_size, name = 'RoBERTa_CNN_1')(model_roberta)
        model_roberta = MaxPooling1D(name='RoBERTa_MaxPolling_1')(model_roberta)
        model_roberta = Flatten(name='RoBERTa_Flatten_1')(model_roberta)
        # model_unigram = Model(inputs=input_unigram, outputs=model_unigram)

        combined = Concatenate(name='Model_Concat')([model_bart, model_roberta])
        combined = Reshape((1, combined.shape[1]), name = 'COMB_Reshape_1')(combined) # reshape to number of units
        combined = Bidirectional(LSTM(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'COMB_BiLSTM_1')(combined)
        combined = Bidirectional(LSTM(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'COMB_BiLSTM_2')(combined)
        combined = Bidirectional(LSTM(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'COMB_BiLSTM_3')(combined)
        combined = Reshape((no_attributes, 1), name = 'COMB_Reshape_2')(combined) # reshape to number of units
        combined = Conv1D(filters = filters, kernel_size=kernel_size, name = 'COMB_CNN_1')(combined)
        combined = MaxPooling1D(name='COMB_MaxPolling_1')(combined)
        combined = Flatten(name='COMB_Flatten_1')(combined)
        output = Dense(units=num_classes, activation = 'softmax', name = 'Model_Output')(combined) #sigmoid #softmax

        # model = Model(inputs=[model_unigram.input, model_bigram.input, model_trigram.input], outputs=output)
        model = Model(inputs=[input_bart, input_roberta], outputs=output, name="BART-RoBERTa-BiLSTM-CNN")
        learning_rate = 0.00000005  # 你可以根据需要调整这个数值
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])


        print(model.summary())
        #plot_model(model, show_shapes=True, dpi=90)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        # mc = ModelCheckpoint(fileName, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

        #history = model.fit(x = [x_vec_train[0], x_vec_train[1]], y = y_vec_train, epochs=epochs_n, verbose=True, validation_data=([x_vec_test[0], x_vec_test[1]], y_vec_test), batch_size=100, callbacks=[es])
        history = model.fit(x = [x_vec_train[0], x_vec_train[1]], y = y_vec_train, epochs=epochs_n, verbose=True, validation_data=([x_vec_test[0], x_vec_test[1]], y_vec_test), batch_size=64)

        y_pred = model.predict([x_vec_test[0], x_vec_test[1]], verbose=False)
        y_p, y_t = evaluate(y_vec_test, y_pred, modelName=model.name, iters=idx)
        end_time = time.time()
        execution_time.append(end_time - start_time)
        print("Time taken to train: ", end_time - start_time)

    print("\n\n========================================\n\n")
    print("BART-RoBERTa-3LSTM-CNN")
    print("Accuracy",np.mean(accuracies), np.std(accuracies))
    print("Precision Micro",np.mean(precisions_micro), np.std(precisions_micro))
    print("Precision Macro",np.mean(precisions_macro), np.std(precisions_macro))
    print("Recall Micro",np.mean(recalls_micro), np.std(recalls_micro))
    print("Recall Macro",np.mean(recalls_macro), np.std(recalls_macro))
    print("Execution Time",np.mean(execution_time), np.std(execution_time))
