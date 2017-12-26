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
from keras.layers import GlobalAveragePooling1D

"""
The code in this file is based on the following source code: 
https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
"""

def LSTM_Sentence_Classifier(numHidden,top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, y_train):
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(numHidden))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, batch_size, epochs)
    return model
    
def LSTM_Dropout_Sentence_Classifier(dropoutRate, numHidden, top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, y_train):
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Dropout(dropoutRate))
    model.add(LSTM(numHidden))
    model.add(Dropout(dropoutRate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, batch_size, epochs)
    return model
    
def CNN_LSTM_Sentence_Classifier(numHidden,top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, y_train):
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
    return model

def CNN_Sentence_Classifier(dropoutRate, top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, y_train):
    seq_length = max_review_length
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Conv1D(64, 3, activation='relu', padding='same', input_shape=(seq_length, 100)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(dropoutRate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print(model.summary())    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    return model

def Stacked_LSTM_Sentence_Classifier(timesteps, numHidden,top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, y_train):
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(numHidden, return_sequences=True,
                   input_shape=(timesteps, max_review_length)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(numHidden, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(numHidden))  # return a single vector of dimension 32
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train,
              batch_size, epochs)
    return model