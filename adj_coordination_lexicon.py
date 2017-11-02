#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:48:18 2017

@author: lixiaodan
"""

import nltk
from nltk.corpus import treebank
import NLP_module 
import math

def expdByAnd(target, extension):
    print(target)
    wsjs = nltk.corpus.treebank.fileids()
    for fileName in wsjs:
        for cursent in treebank.tagged_sents(fileName):
            for i in range(len(cursent)):
                if i == 0 or i == len(cursent) - 1:
                    continue
                if NLP_module.strCmp(cursent[i][0], 'and'):
                    if NLP_module.strCmp(cursent[i - 1][0], target):
                        if NLP_module.strCmp(cursent[i + 1][1], "JJ"):
                            extension.append(cursent[i + 1][0])
                            print(cursent[i + 1][0])
                    ## ?? and good
                    if NLP_module.strCmp(cursent[i + 1][0], target):
                        if NLP_module.strCmp(cursent[i - 1][1], "JJ"):
                            extension.append(cursent[i - 1][0])
                            print(cursent[i - 1][0])
    return extension

def expdByBut(target):
    print(target)
    extension = list()
    wsjs = nltk.corpus.treebank.fileids()
    for fileName in wsjs:
        for cursent in treebank.tagged_sents(fileName):
            for i in range(len(cursent)):
                if i == 0 or i == len(cursent) - 1:
                    continue
                if NLP_module.strCmp(cursent[i][0], 'but'):
                    if NLP_module.strCmp(cursent[i - 1][0], target):
                        if NLP_module.strCmp(cursent[i + 1][1], "JJ"):
                            extension.append(cursent[i + 1][0])
                            print(cursent[i + 1][0])
                    ## ?? and good
                    if NLP_module.strCmp(cursent[i + 1][0], target):
                        if NLP_module.strCmp(cursent[i - 1][1], "JJ"):
                            extension.append(cursent[i - 1][0])
                            print(cursent[i - 1][0])
    return extension

def expdByNegPref(target, prefixes):
    result = list()
    for i in range(len(prefixes)):
        curPref = prefixes[i]
        newWord = curPref + target
        wsjs = nltk.corpus.treebank.fileids()
        for fileName in wsjs:
            for curWds in treebank.words(fileName):
                if newWord in curWds:
                     result.append(newWord)
    return result

def expdByNegPostf(target, postfixes):
    result = list()
    for i in range(len(postfixes)):
        curPost = postfixes[i]
        newWord = target + curPost
        wsjs = nltk.corpus.treebank.fileids()
        for fileName in wsjs:
            for curWds in treebank.words(fileName):
                if newWord in curWds:
                     result.append(newWord)
    return result

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

"""
#################### using seed words and adjective coordination ################
prefixes = ["un", "im"]
postfixes = ["less"]
for wd in posseeds:
    extensionP = expdByAnd(wd, extensionP)
    ext_neg = expdByBut(wd)
    ext_neg_pref = expdByNegPref(wd, prefixes)
    ext_neg_post = expdByNegPostf(wd, postfixes)
print(extensionP)

for wd in negseeds:
    extensionN = expdByAnd(wd, extensionN)
    ext_pos = expdByBut(wd)
    ext_pos_pref = expdByNegPref(wd, prefixes)
    ext_pos_post = expdByNegPostf(wd, postfixes)
print(extensionN)

extensionP = extensionP + ext_pos + ext_pos_post + ext_pos_pref
extensionN = extensionN + ext_neg + ext_neg_post + ext_neg_pref
"""

### another way to get and, but adjs.
wsjs = nltk.corpus.treebank.fileids()
but_hash = dict()
and_hash = dict()
but_pairs = list()
and_pairs = list()
for fileName in wsjs:
    for cursent in treebank.tagged_sents(fileName):
        for i in range(len(cursent)):
            if i == 0 or i == len(cursent) - 1:
                continue
            pair = list()
            if NLP_module.strCmp(cursent[i][0], 'but'):
                if NLP_module.strCmp(cursent[i - 1][1], 'JJ') and NLP_module.strCmp(cursent[i + 1][1], 'JJ'): 
                    pair.append(cursent[i - 1][0])
                    pair.append(cursent[i + 1][0])
                    but_pairs.append(pair)
                    if cursent[i - 1][0] in but_hash.keys():
                        but_hash.get(cursent[i - 1][0]).add(cursent[i + 1][0])
                    else:
                        buts = set()
                        buts.add(cursent[i + 1][0])
                        but_hash[cursent[i - 1][0]] = buts
                    if cursent[i + 1][0] in but_hash.keys():
                        but_hash.get(cursent[i + 1][0]).add(cursent[i - 1][0])
                    else:
                        buts = set()
                        buts.add(cursent[i - 1][0])
                        but_hash[cursent[i + 1][0]] = buts
            pair = list()
            if NLP_module.strCmp(cursent[i][0], 'and'):
                if NLP_module.strCmp(cursent[i - 1][1], 'JJ') and NLP_module.strCmp(cursent[i + 1][1], 'JJ'): 
                    pair.append(cursent[i - 1][0])
                    pair.append(cursent[i + 1][0])
                    and_pairs.append(pair)
                    if cursent[i - 1][0] in and_hash.keys():
                        and_hash.get(cursent[i - 1][0]).add(cursent[i + 1][0])
                    else:
                        ands = set()
                        ands.add(cursent[i + 1][0])
                        and_hash[cursent[i - 1][0]] = ands
                    if cursent[i + 1][0] in and_hash.keys():
                        and_hash.get(cursent[i + 1][0]).add(cursent[i - 1][0])
                    else:
                        ands = set()
                        ands.add(cursent[i - 1][0])
                        and_hash[cursent[i + 1][0]] = ands
print("but_hash")
print(but_hash)
print("and_hash")
print(and_hash)            

for wd in posseeds:
    if wd in and_hash.keys():
        extensionP = extensionP + list(and_hash.get(wd))
        extensionP = list(set(extensionP))
    if wd in but_hash.keys():
        extensionN = extensionN + list(but_hash.get(wd))
        extensionN = list(set(extensionN))
        
for wd in negseeds:
    if wd in and_hash.keys():
        extensionN = extensionN + list(and_hash.get(wd))
        extensionN = list(set(extensionN))
    if wd in but_hash.keys():
        extensionP = extensionP + list(but_hash.get(wd))
        extensionP = list(set(extensionP))