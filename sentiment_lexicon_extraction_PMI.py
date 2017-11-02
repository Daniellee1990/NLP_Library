#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 09:40:11 2017

@author: lixiaodan
"""
import nltk
from nltk.corpus import treebank
import NLP_module 
import math

posseeds = list()
negseeds = list()

label_path = '/Users/lixiaodan/Desktop/ece590/negative-words.txt'

with open(label_path, 'r', encoding = "ISO-8859-1") as input_file:
    for line in input_file:
        negseeds.append(line.split('\n')[0])
    input_file.close()

try:
    label_path = '/Users/lixiaodan/Desktop/ece590/positive-words.txt'
    infile = open(label_path, 'r')
    for line in infile:
        posseeds.append(line.split('\n')[0])
    infile.close()
except UnicodeDecodeError:
    print("Positive Unicode error")

extensionP = list()
extensionN = list()

### Pointwise mutual information #####
## Step 1: extract phrases ##
target = 'unusually'
#seed = 'easy'
k = 2      
phrase_pattern = list()
wsjs = nltk.corpus.treebank.fileids()
N = 0
for filename in wsjs:
    for cursent in treebank.tagged_sents(filename):
        N  = N + len(cursent)
        for i in range(len(cursent)):
            if NLP_module.strCmp(cursent[i][1], 'JJ') == True:
                k = 1
                while (i + k < len(cursent)):
                    if NLP_module.strCmp(cursent[i + k][1], 'NN') == True or NLP_module.strCmp(cursent[i + k][1], 'NNS') == True:
                        pair = list()
                        pair.append(cursent[i])
                        pair.append(cursent[i + k])
                        phrase_pattern.append(pair)
                    k = k + 1
            if NLP_module.strCmp(cursent[i][1], 'RB') == True or NLP_module.strCmp(cursent[i][1], 'RBR') == True or NLP_module.strCmp(cursent[i][1], 'RBS') == True:
                k = 1
                while (i + k < len(cursent)):
                    if NLP_module.strCmp(cursent[i + k][1], 'VB') == True or NLP_module.strCmp(cursent[i + k][1], 'VBD') == True or NLP_module.strCmp(cursent[i + k][1], 'VBN') == True or NLP_module.strCmp(cursent[i + k][1], 'VBG') == True:
                        pair = list()
                        pair.append(cursent[i])
                        pair.append(cursent[i + k])
                        phrase_pattern.append(pair)
                    k = k + 1
            if NLP_module.strCmp(cursent[i][1], 'RB') == True or NLP_module.strCmp(cursent[i][1], 'RBR') == True or NLP_module.strCmp(cursent[i][1], 'RBS') == True or NLP_module.strCmp(cursent[i][1], 'JJ') == True or NLP_module.strCmp(cursent[i][1], 'NN') == True or NLP_module.strCmp(cursent[i][1], 'NNS') == True:
                k = 1
                while (i + k < len(cursent)):
                    if NLP_module.strCmp(cursent[i + k][1], 'JJ') == True:
                        if i + k == len(cursent) - 1:
                            pair = list()
                            pair.append(cursent[i])
                            pair.append(cursent[i + k])
                            phrase_pattern.append(pair)
                        elif NLP_module.strCmp(cursent[i + k + 1][1], 'NN') == False and NLP_module.strCmp(cursent[i + k + 1][1], 'NNS') == False:
                            pair = list()
                            pair.append(cursent[i])
                            pair.append(cursent[i + k])
                            phrase_pattern.append(pair)
                    k = k + 1

PosPMI = 0 
for seed in posseeds:
    concur = 0
    wdhit = 0
    seedhit = 0 
    for phrase in phrase_pattern:
        if NLP_module.strCmp(phrase[0][0], target) == True and NLP_module.strCmp(phrase[1][0], seed) == True:
            concur = concur + 1
        if NLP_module.strCmp(phrase[0][0], seed) == True and NLP_module.strCmp(phrase[1][0], target) == True:
            concur = concur + 1
        if NLP_module.strCmp(phrase[0][0], seed) == True and NLP_module.strCmp(phrase[1][0], seed) == True:
            seedhit = seedhit + 1
        if NLP_module.strCmp(phrase[0][0], target) == True and NLP_module.strCmp(phrase[1][0], target) == True:
            wdhit = wdhit + 1
    numer = (1.0 / (k * N)) * concur
    denom = (1.0 / N) * wdhit * (1.0 / N) * seedhit
    PMI = 0
    if numer != 0 and denom != 0:
        PMI = math.log2(numer / denom)
    PosPMI = PosPMI + PMI
    if PMI != 0:
        print(seed)