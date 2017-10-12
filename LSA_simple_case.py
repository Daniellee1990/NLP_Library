#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:17:14 2017

@author: lixiaodan
"""

from nltk.corpus import brown
import numpy as np
import math

brown_sci = brown.words(categories = 'science_fiction')
brown_humor = brown.words(categories = 'humor')
brown_news = brown.words(categories = 'news')
brown_religion = brown.words(categories = 'religion')
brown_hobbies = brown.words(categories = 'hobbies')

unique_sci = set(brown_sci)
unique_humor = set(brown_humor)
unique_news = set(brown_news)
unique_rel = set(brown_religion)
unique_hb = set(brown_hobbies)

word2num_sci = dict()
word2num_humor = dict()
word2num_news = dict()
word2num_rel = dict()
word2num_hb = dict()
uniques = set.union( unique_sci, unique_humor, unique_news , unique_rel, unique_hb)
rows = list(uniques)

for word in brown_sci:
    if word in word2num_sci.keys():
        word2num_sci[word] = word2num_sci[word] + 1
    else:
        word2num_sci[word] = 1
        
for word in brown_humor:
    if word in word2num_humor.keys():
        word2num_humor[word] = word2num_humor[word] + 1
    else:
        word2num_humor[word] = 1
        
for word in brown_news:
    if word in word2num_news.keys():
        word2num_news[word] = word2num_news[word] + 1
    else:
        word2num_news[word] = 1
        
for word in brown_religion:
    if word in word2num_rel.keys():
        word2num_rel[word] = word2num_rel[word] + 1
    else:
        word2num_rel[word] = 1        

for word in brown_hobbies:
    if word in word2num_hb.keys():
        word2num_hb[word] = word2num_hb[word] + 1
    else:
        word2num_hb[word] = 1 
        
numWord = len(uniques)
numDoc = 5
# create matrix
term_doc = np.zeros((numWord,numDoc))
rowName = list(uniques)

# first column
for j in range(len(rowName)):
    if rowName[j] in word2num_sci.keys():
        tp = word2num_sci.get(rowName[j])
    else:
        tp = 0
    term_doc[j][0] = tp
    
# second column
for j in range(len(rowName)):
    if rowName[j] in word2num_humor.keys():
        tp = word2num_humor.get(rowName[j])
    else:
        tp = 0
    term_doc[j][1] = tp

# third column
for j in range(len(rowName)):
    if rowName[j] in word2num_news.keys():
        tp = word2num_news.get(rowName[j])
    else:
        tp = 0
    term_doc[j][2] = tp

# fourth column
for j in range(len(rowName)):
    if rowName[j] in word2num_rel.keys():
        tp = word2num_rel.get(rowName[j])
    else:
        tp = 0
    term_doc[j][3] = tp
    
for j in range(len(rowName)):
    if rowName[j] in word2num_hb.keys():
        tp = word2num_hb.get(rowName[j])
    else:
        tp = 0
    term_doc[j][4] = tp

term_doc_normed = term_doc / term_doc.sum(axis=0)

for i in range(len(rowName)):
    tt = 0
    for j in range(numDoc):
        if term_doc_normed[i][j] != 0:
            tt = tt + 1
    if tt == 0:
        continue
    idf = math.log(numDoc * 1.0 / tt)
    for k in range(numDoc):
        term_doc_normed[i][k] = term_doc_normed[i][k] * idf
print(term_doc_normed)
U, s, V = np.linalg.svd(term_doc_normed)
print(s)

# get first k dimension
k = 3
Uk = np.matrix(U[:, 0:k])
Vk = np.matrix(V[0:k, :])
tp = s[0:k]
sk = np.diag(tp)
reduced_matrix = Uk
