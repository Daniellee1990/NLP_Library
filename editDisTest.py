#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:42:37 2017

@author: lixiaodan
"""

import NLP_module
import pandas as pd
import numpy as np

"""
str1 = "daniel"
str2 = "dadafasdfasdfasdfsa"
print(NLP_module.minEditDisCost(str1, str2))

str1 = "daniel"
str2 = "danopt"
print(NLP_module.minEditDisCost(str1, str2))

str1 = "daniel_dan"
str2 = "d2n"
print(NLP_module.minEditDis(str1, str2))
print(NLP_module.minEditDisCost(str1, str2))

str1 = "warren"
str2 = "warrant"
print("warren -> warrant")
print(NLP_module.minEditDis(str1, str2))

str1 = "phiosophy"
str2 = "philanthropy"
print("phiosophy -> philanthropy")
print(NLP_module.minEditDisWithSwap(str1, str2))

str1 = "phiosophy"
str2 = "phiasopyh"
print("phiosophy -> phiasopyh")
print(NLP_module.minEditDisWithSwap(str1, str2))
"""

df = pd.read_csv('/Users/lixiaodan/Desktop/ece590/DGA/top-1m.csv', header=None)
goodNames = df[1]
gDomains = list()
for i in range(len(goodNames)):
    domainTp = goodNames[i].split('.')[0]
    gDomains.append(domainTp)
gDomains = gDomains[:10000]
# get malicious letters distribution
file = open('/Users/lixiaodan/Desktop/ece590/DGA/conficker.txt', 'r')
dgas = list()
for line in file:
    dga = line.split('.')[0]
    dgas.append(dga) 
    
"""
minDisDga = list()    
for dga in dgas:
    minDis = 40
    simWord = ""
    for benDom in gDomains:
        dis = NLP_module.minEditDis(benDom, dga)
        if dis < minDis:
            minDis = dis
            simWord = benDom
    tp = list()
    tp.append(dga)
    tp.append(dis)
    tp.append(benDom)
    minDisDga.append(tp)
dgameanDis = np.mean(minDisDga)
dgamin = np.min(minDisDga)
dgamax = np.max(minDisDga) 
print(dgameanDis)
print(dgamin)
print(dgamax) 
"""

minDises = list()
for str1 in gDomains:
    minD = 100
    for str2 in gDomains:
        if str1 == str2:
            continue
        dis = NLP_module.minEditDis(str1, str2)
        if dis < minD:
            minD = dis
    minDises.append(minD)
benmeanDis = np.mean(minDises)
dismin = np.min(minDises)
dismax = np.max(minDises)
print(benmeanDis)
print(dismin)
print(dismax)     


