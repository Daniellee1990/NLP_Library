#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:54:55 2017

@author: lixiaodan
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding

"""
The code in this file is based on the following source code: 
https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
"""

def LSTM_Sentence_Classifier(numHidden,top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(numHidden))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, batch_size, epochs)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
def LSTM_Dropout_Sentence_Classifier(dropoutRate, numHidden, top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Dropout(dropoutRate))
    model.add(LSTM(numHidden))
    model.add(Dropout(dropoutRate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, batch_size, epochs)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

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
    model.fit(X_train, y_train, batch_size, epochs)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))