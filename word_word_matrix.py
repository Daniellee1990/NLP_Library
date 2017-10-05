#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 22:02:46 2017

@author: lixiaodan
"""

from nltk.corpus import brown
import numpy as np
import math

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
        #print("current word is", curwd)
        rownum = wdList.index(curwd)
        #print("rownum is", rownum)
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

addN = 2
# get Pointwise Mutual Information(PMI)
word_word_pro = np.zeros((len(word_names), len(word_names)))
for i in range(len(word_names)):
    for j in range(len(word_names)):
        word_word_pro[i][j] = (word_word[i][j] + addN) * 1.0 / (total + total * addN)
wordFre = np.sum(word_word_pro, axis = 1)
contextFre = np.sum(word_word_pro, axis = 0)

#get whichever the string1 and string2 and get their PMI        
word = "effective"
contex = "policies"
rownum = word_names.index(word)
colnum = word_names.index(contex)
pwc = word_word_pro[rownum][colnum]
pc = contextFre[colnum]
pw = wordFre[rownum]
if pwc == 0:
    PMI = 0
else:
    PMI = math.log2(pwc * 1.0 / (pw * pc))
    if PMI < 0:
        PMI = 0
#print(PMI)


    