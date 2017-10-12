#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:09:05 2017

@author: lixiaodan
"""

from nltk.corpus import brown
import numpy as np
import math

def gettf_idf(term_doc):    
    term_doc_normed = term_doc / term_doc.sum(axis=0)
    rowNum = term_doc.shape[0]
    docNum = term_doc.shape[1]
    for i in range(rowNum):
        tt = 0
        for j in range(docNum):
            if term_doc_normed[i][j] != 0:
                tt = tt + 1
        if tt == 0:
            continue
        idf = math.log(docNum * 1.0 / tt)
        for k in range(docNum):
            term_doc_normed[i][k] = term_doc_normed[i][k] * idf
    return term_doc_normed

docNum = len(brown.categories())
uniqueSet = set()
for category in brown.categories():
    curWords = brown.words(categories=category)
    curSet = set(curWords)
    uniqueSet = set.union(uniqueSet, curSet)
numWords = len(uniqueSet)

# create word-term matrix
term_doc = np.zeros((numWords,docNum))
rowName = list(uniqueSet)
cates = list(brown.categories())

# fill every entry in word-term matrix
for k in range(docNum):
    category = cates[k] 
    curWds = brown.words(categories = category)
    word2num = dict()
    # create hashmap for the words
    for word in curWds:
        if word in word2num:
            word2num[word] = word2num[word] + 1
        else:
            word2num[word] = 1
    for i in range(len(rowName)):
        if rowName[i] in word2num.keys():
            tp = word2num.get(rowName[i])
        else:
            tp = 0
        term_doc[i][k] = tp

# get the tf-idf
term_doc_tfidf = gettf_idf(term_doc)

# get word_term for LSA
fre_term_doc = term_doc / term_doc.sum(axis=0)
local_term_doc = np.zeros((numWords, docNum))
for i in range(numWords):
    for j in range(docNum):
        if fre_term_doc[i][j] == 0:
            continue
        local_term_doc[i][j] = np.log(fre_term_doc[i][j]) + 1
        
global_term_doc = np.zeros((numWords, docNum))
for i in range(numWords):
    # get sum
    sum = 0
    for j in range(docNum):
        if fre_term_doc[i][j] == 0:
            continue
        sum = sum + fre_term_doc[i][j] * np.log(fre_term_doc[i][j])
        
    para = 1 + sum * 1.0 / np.log(docNum)
    # set global weight for cell(i, k)
    for k in range(docNum):
        global_term_doc[i][k] = para
"""
U, s, V = np.linalg.svd(term_doc_normed)
print(s)
"""