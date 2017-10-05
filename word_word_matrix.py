#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 22:02:46 2017

@author: lixiaodan
"""

from nltk.corpus import brown
import numpy as np

def getWordWordMatrix(words, neighbor):
    wdList = list()
    for wd in words:
         wdList.append(wd)
    wdSet = set(wdList)
    wdList = list(wdSet)
    word_word = np.zeros((len(wdSet), len(wdSet)))
    neighbor = 3 
    # find the words following the word with the distance neighbor points out.
    for i in range(len(words) - 1):
        curwd = words[i]
        rownum = wdList.index(curwd)
        if i >= len(words) - neighbor:
            for k in range(i + 1, len(words)):
                curNb = words[k]
                colnum = wdList.index(curNb)
                word_word[rownum][colnum] = word_word[rownum][colnum] + 1
        else:
            for j in range(1, neighbor + 1):
                curNb = words[i + j]
                colnum = wdList.index(curNb)
                word_word[rownum][colnum] = word_word[rownum][colnum] + 1

    # find the words before the word with distance neighbor points out.
    for i in range(len(words) - 1, len(words) - neighbor, -1):
        curwd = words[i]
        rownum = wdList.index(curwd)
        for j in range(1, neighbor + 1):
            curNb = words[i - j]
            colnum = wdList.index(curNb)
            word_word[rownum][colnum] = word_word[rownum][colnum] + 1
    for i in range(1, neighbor + 1):
        curwd = words[i]
        rownum = wdList.index(curwd)
        for j in range(0, i):
            curNb = words[j]
            colnum = wdList.index(curNb)
            word_word[rownum][colnum] = word_word[rownum][colnum] + 1
    return word_word, wdList

words = brown.words()[:500]
total = len(words)
neighbor = 3
word_word, word_names = getWordWordMatrix(words, neighbor)

for i in range(word_word.shape[0]):
    for j in range(word_word.shape[1]):
        if word_word[i][j] != 0:
            print(word_word[i][j])

# get Pointwise Mutual Information(PMI)
word_word_pro = np.zeros((len(word_names), len(word_names)))
for i in range(len(word_names)):
    for j in range(len(word_names)):
        word_word_pro[i][j] = word_word[i][j] * 1.0 / total
wordFre = np.sum(word_word_pro, axis=1)
print(wordFre)

#get whichever the string1 and string2 and get their PMI        
str1 = "daniel"
str2 = "xiaodan"

    