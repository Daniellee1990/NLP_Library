#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 15:01:46 2017

@author: lixiaodan
"""

import numpy as np
import email
#import tensorflow as tf
import html2text
import re
import os
import NLP_module
import collections
from nltk.corpus import stopwords
#from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

"""
https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
Simple LSTM for Sequence Classification
"""

# create the model
def LSTM_Dropout_Sentence_Classifier(numHidden,top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(numHidden))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs, batch_size)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# Build word vocabulary function
def build_vocab(text, top_words):
    word_counts = collections.Counter(text.split(' '))
    word_counts = word_counts.most_common(top_words - 1)
    vocab_to_ix_dict = dict()
    # Add unknown key --> 0 index
    vocab_to_ix_dict['unknown']=0
    index = 1 
    for item in word_counts:
        vocab_to_ix_dict[item[0]] = index
        index = index + 1
    # Create index --> vocab mapping
    ix_to_vocab_dict = {val:key for key,val in vocab_to_ix_dict.items()}
    
    return(ix_to_vocab_dict, vocab_to_ix_dict)

# Create a text cleaning function
def clean_text(text_string): 
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.escape_snob = True
    txt = h.handle(text_string)
    txt = txt.lower()
    p = re.compile('\W+')
    splits  = p.split(txt)
    result = ""
    start_to_parse = False
    for word in splits:
        if word.find("_") == 0:
            word = word[1:]
        if NLP_module.strCmp(word, "Date"):
            start_to_parse = True
        if start_to_parse == False:
            continue
        found = re.search('[0-9]+', word)
        if found != None:
            continue
        stop_wds = stopwords.words("english")
        stop_wds.extend(["email", "www", "com", "http", "html", "gif", "smtp", "sender", "received", "zzzz", "yyyy","localhost", "org", "esmtp", "debian", "return", "path"])
        if word in stop_wds:
            continue
        if len(word) <= 2 or len(word) >= 10:
            continue
        result = result + " " + word
    text_string = result.lower()
    return(text_string)

labels = dict()
label_path = '/Users/lixiaodan/Desktop/ece590/CSDMC2010_SPAM/CSDMC2010_SPAM/SPAMTrain.label'
infile = open(label_path,'r')
label_List = list()
for line in infile:
    tp = line.split(" ")[1]
    eml_name = tp.split("\n")[0]
    labels[eml_name] = line.split(" ")[0]
    label_List.append(line.split(" ")[0])
infile.close()

path = '/Users/lixiaodan/Desktop/ece590/CSDMC2010_SPAM/CSDMC2010_SPAM/training_new'
listing = os.listdir(path)
listing = listing

fail_IO = list()
gd_cnt = 0
bad_cnt = 0
text_target = list()
text_data_train = list()
texts = ""

for i in range(len(listing)):
    fle = listing[i]
    if str.lower(fle[-3:])=="eml":
        try:
            msg = email.message_from_file(open(path + '/' + fle))
            strs = msg.as_string()
            cleantext = clean_text(strs)
            text_data_train.append(cleantext)
            texts = texts + cleantext
            if labels[fle] == "1":
                gd_cnt = gd_cnt + 1
                text_target.append(1)
            else:
                bad_cnt = bad_cnt + 1
                text_target.append(0)
        except UnicodeDecodeError:
            fail_IO.append(fle)
            continue
min_word_freq = 100
max_review_length = 500
ix2word, word2ix = build_vocab(texts, top_words)
text_processed = list()
# Convert text to word vectors
for s_text in text_data_train:
    s_text_words = s_text.split(' ')
    s_text_ix = list()
    for ix, x in enumerate(s_text_words):
        try:
            s_text_ix.append(word2ix[x])
        except:
            s_text_ix.append(0)
    cur_text_ix = s_text_ix[0:max_review_length]
    if len(cur_text_ix) < 500:
        for i in range(500 - len(cur_text_ix)):
            cur_text_ix.append(0)
    text_processed.append(cur_text_ix)
    
# Shuffle and split data
text_processed = np.array(text_processed)
text_data_target = np.array(text_target)
#text_data_target = np.array([1 if x=='ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

# Split train/test set
ix_cutoff = int(len(y_shuffled)*0.80)
X_train, X_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
#vocab_size = len(vocab_processor.vocabulary_)
#print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
embedding_vector_length = 32
numHidden = 100
epochs = 50
batch_size=64

# create the model
LSTM_Dropout_Sentence_Classifier(numHidden,top_words, embedding_vector_length, max_review_length, epochs, batch_size, X_train, X_test, y_train, y_test)