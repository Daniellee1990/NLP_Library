#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:26:58 2017

@author: lixiaodan
"""

import NLP_module
import pandas as pd

df = pd.read_csv('/Users/lixiaodan/Desktop/ece590/DGA/top-1m.csv', header=None)
goodNames = df[1]
gDomains = list()
for i in range(len(goodNames)):
    domainTp = goodNames[i].split('.')[0]
    gDomains.append(domainTp)
gDomains = gDomains[:5000]    
bigramLists = NLP_module.getBiList(gDomains)

# get malicious letters distribution
file = open('/Users/lixiaodan/Desktop/ece590/DGA/conficker.txt', 'r')
dgas = list()
for line in file:
    dga = line.split('.')[0]
    dgas.append(dga)

dgaJI = list()
for dga in dgas:
    dgaBis = list()
    tp = ''
    for i in range(len(dga)):
        tp = tp + dga[i]
        if i >= 1:
            dgaBis.append(tp)
            tp = tp[1:]
    sdga = set(dgaBis)
    closeDm = ''
    maxJI = 0
    for dom in bigramLists.keys():
        domBis = set(bigramLists.get(dom))
        universe = sdga.union(domBis)
        overlap = sdga.intersection(domBis)
        tp = len(overlap) * 1.0 / len(universe)
        if tp > maxJI:
            maxJI = tp
            closeDm = dom
    tp = list()
    tp.append(dga)
    tp.append(closeDm)
    tp.append(maxJI)
    dgaJI.append(tp)

domJI = list()
for dom in bigramLists.keys():
    curBis = bigramLists.get(dom) 
    maxJI = 0
    closeDm = ''
    for other in bigramLists.keys():
        if dom == other:
            continue
        curBisSet = set(curBis)
        otherSet = set(bigramLists.get(other))
        uni = curBisSet | otherSet
        over = curBisSet & otherSet
        tp = len(over) * 1.0 / len(uni)
        if tp > maxJI:
            maxJI = tp
            closeDm = other
    tp = list()
    tp.append(dom)
    tp.append(closeDm)
    tp.append(maxJI)
    domJI.append(tp)        