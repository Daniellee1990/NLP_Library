#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 15:50:40 2017

@author: lixiaodan
"""

# LSTM with Dropout for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

def LSTM_Dropout_Sentence_Classifier(dropoutRate,numHidden,top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Dropout(dropoutRate))
    model.add(LSTM(numHidden))
    model.add(Dropout(dropoutRate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs, batch_size)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vector_length = 32
dropoutRate = 0.2
numHidden = 100
epochs=3
batch_size=64

LSTM_Dropout_Sentence_Classifier(dropoutRate,numHidden,top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, X_test, y_train, y_test)