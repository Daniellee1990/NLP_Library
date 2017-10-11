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

def cosineSimilarity(wordVec1, wordVec2):
    vec1_mode = 0
    vec2_mode = 0
    dot_product = 0
    for i in range(len(wordVec1)):
        vec1_mode = vec1_mode + wordVec1[i] * wordVec1[i]
        vec2_mode = vec2_mode + wordVec2[i] * wordVec2[i]
        dot_product = dot_product + wordVec1[i] * wordVec2[i]
    cos = dot_product * 1.0 / (math.sqrt(vec1_mode) * math.sqrt(vec2_mode) * 1.0) 
    return cos

def JaccSim(wordVec1, wordVec2):
    numerator = 0.0
    denomin = 0.0
    for i in range(len(wordVec1)):
        min = wordVec1[i]
        if wordVec2[i] < min:
            min = wordVec2[i]
        max = wordVec1[i]
        if wordVec2[i] > max:
            max = wordVec2[2]
        numerator = numerator + min
        denomin = denomin + max
    return numerator * 1.0 / denomin

def DiceSim(wordVec1, wordVec2):
    numer = 0
    denom = 0
    for i in range(len(wordVec1)):
        min = wordVec1[i]
        if min > wordVec2[i]:
            min = wordVec2[i]
        numer = numer + min
        denom = denom + wordVec1[i] + wordVec2[i]
    simDice = numer * 2.0 / denom
    return simDice

# get Pointwise Mutual Information(PMI)
def getPMI(word, context, word_names, addN):
    word_word_pro = np.zeros((len(word_names), len(word_names)))
    for i in range(len(word_names)):
        for j in range(len(word_names)):
            # add-n smooth method
            word_word_pro[i][j] = (word_word[i][j] + addN) * 1.0 / (total + total * addN)
    wordFre = np.sum(word_word_pro, axis = 1)
    contextFre = np.sum(word_word_pro, axis = 0)
    #get whichever the string1 and string2 and get their PMI        
    rownum = word_names.index(word)
    colnum = word_names.index(context)
    pwc = word_word_pro[rownum][colnum]
    pc = contextFre[colnum]
    pw = wordFre[rownum]
    if pwc == 0:
        PMI = 0
    else:
        PMI = math.log2(pwc * 1.0 / (pw * pc))
        if PMI < 0:
            PMI = 0
    return PMI

words = brown.words()[:2000]
total = len(words)
neighbor = 3
word_word, word_names = getWordWordMatrix(words, neighbor)
#print(word_word)

addN = 2
word = "effective"
context = "policies"
PMI = getPMI(word, context, word_names, addN)

# get cosine similarity
wordVec1 = word_word[55,:]
word1 = word_names[55]
wordVec2 = word_word[56,:]
word2 = word_names[56]
cos = cosineSimilarity(wordVec1, wordVec2)

# jaccard result
jacc = JaccSim(wordVec1, wordVec2)
print(jacc)

"""
print("word1 is :")
print(word1)
print("word2 is :")
print(word2)
print("cosine is :")
print(cos)
"""

similarWords = list()
for i in range(word_word.shape[0]):
    curVec = word_word[i,:]
    curWord = word_names[i]
    maxcos = 0
    simword = ''
    for j in range(word_word.shape[0]):
        if i == j:
            continue
        otherVec = word_word[j,:]
        otherWord = word_names[j]
        curcos = cosineSimilarity(curVec, otherVec)
        if curcos > maxcos:
            maxcos = curcos
            simword = otherWord
    tp = list()
    tp.append(curWord)
    tp.append(simword)
    tp.append(maxcos)
    similarWords.append(tp)
    
for tp in similarWords:
    print(tp)

# Dice measure 
wordVec1 = word_word[55, :]
wordVec2 = word_word[56, :]
simDice = DiceSim(wordVec1, wordVec2)
print("sim dice")
print(simDice)
    
