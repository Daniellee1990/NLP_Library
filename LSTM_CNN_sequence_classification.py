#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 15:51:56 2017

@author: lixiaodan
"""

"""
https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
LSTM and Convolutional Neural Network For Sequence Classification
"""

def CNN_LSTM_Sentence_Classifier(numHidden,top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, X_test, y_train, y_test):
    # create the model
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(numHidden))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs, batch_size)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
embedding_vector_length = 32
numHidden = 100
epochs = 3
batch_size = 64
#train CNN and LSTM model
CNN_LSTM_Sentence_Classifier(numHidden,top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, X_test, y_train, y_test)