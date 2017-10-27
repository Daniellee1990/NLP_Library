#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:34:38 2017

@author: lixiaodan
"""

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

def expandSeedsBySyn(seeds, radius):    
    cur_seeds = seeds
    # expand word set
    for i in range(radius):
        curSysSets = list()
        for i in range(len(cur_seeds)):
            cur_sd = cur_seeds[i]
            synsets = wn.synsets(cur_sd)
            for syn in synsets:
                wd_name = syn.name().split(".")[0]
                curSysSets.append(wd_name)
        seeds = seeds + curSysSets
        cur_seeds = curSysSets
    seeds = list(set(seeds))
    return seeds

def expandSeedsByAnt(seeds, negseeds):
    cur_seeds = negseeds
    # expand word set
    antonyms = list()
    for i in range(len(cur_seeds)):
        cur_sd = cur_seeds[i]
        synsets = wn.synsets(cur_sd)
        for syn in synsets:
            for l in syn.lemmas():
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())    
    seeds = seeds + antonyms
    seeds = list(set(seeds))
    return seeds
     
"""
def builSentimentLexicon(posseeds, negseeds, radius):
    poslex = posseeds
    poslex = expandSeedsBySyn(poslex, radius)
    neglex = negseeds
    neglex = expandSeedsBySyn(neglex, radius)
    return poslex, neglex
"""

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

poslex = posseeds
neglex = negseeds
radius = 3
print(len(poslex))
poslex = expandSeedsBySyn(poslex, radius)
print(len(poslex))
poslex = expandSeedsByAnt(poslex, neglex)
print(len(poslex))

print(len(neglex))
neglex = expandSeedsBySyn(neglex, radius)
print(len(neglex))
neglex = expandSeedsByAnt(neglex, poslex)
print(len(neglex))