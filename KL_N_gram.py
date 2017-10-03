# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import NLP_module

df = pd.read_csv('/Users/lixiaodan/Desktop/ece590/DGA/top-1m.csv', header=None)
goodNames = df[1]
gDomains = list()
for i in range(len(goodNames)):
    domainTp = goodNames[i].split('.')[0]
    gDomains.append(domainTp)

trainGDom = gDomains[0:980000]
testGDom = gDomains[980000:]

# train benigh letters distibution to get bunigram 
bunigram = NLP_module.getUnigram(trainGDom)
    
# train benigh strings to get bbigram
bbigram = NLP_module.getBigram(trainGDom)
        
# get malicious letters distribution
file = open('/Users/lixiaodan/Desktop/ece590/DGA/conficker.txt', 'r')
dgas = list()
for line in file:
    dga = line.split('.')[0]
    dgas.append(dga)
    
mtotal = 0
trainDgas = dgas[0:80000]
testDgas = dgas[80000:]

munigram = NLP_module.getUnigram(trainDgas)

# train malicious strings to get mbigram
mbigram = NLP_module.getBigram(trainDgas)

# 1 - dga, 0 - valid domain
# testing benign dataset
gDomByUni = list()
gDomByBi = list()
for gdDom in testGDom:
    KLdisUni = NLP_module.getUniKLdis(bunigram, munigram, gdDom)
    if KLdisUni <= 0:
        gDomByUni.append(1)
    else :
        gDomByUni.append(0)

    KLdisBi = NLP_module.getBiKLdis(bbigram, mbigram, gdDom)
    if KLdisBi <= 0:
        gDomByBi.append(1)
    else:
        gDomByBi.append(0)

dgaByUni = list()
dgaByBi = list()
for dga in testDgas: 
    KLdisUni = NLP_module.getUniKLdis(bbigram, mbigram, dga)
    if KLdisUni <= 0:
        dgaByUni.append(1)
    else :
        dgaByUni.append(0)

    KLdisBi = NLP_module.getBiKLdis(bbigram, mbigram, dga)
    if KLdisBi <= 0:
        dgaByBi.append(1)
    else:
        dgaByBi.append(0)

y_label = [0] * 20000 + [1] * 20000
resultByUni = gDomByUni + dgaByUni
resultByBi = gDomByBi + dgaByBi

NLP_module.plotRoc(resultByUni, y_label)
NLP_module.plotRoc(resultByBi, y_label)

