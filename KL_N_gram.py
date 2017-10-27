# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import pandas as pd
import NLP_module

df = pd.read_csv('/Users/lixiaodan/NLPkit/DataSet/top-1m.csv', header=None)
goodNames = df[1]
gDomains = list()
for i in range(len(goodNames)):
    domainTp = goodNames[i].split('.')[0]
    gDomains.append(domainTp)

trainGDom = gDomains[0:980000]
testGDom = gDomains[980000:]

# train benigh letters distibution to get bunigram 
bunigramPro = NLP_module.getUnigramPro(trainGDom)
    
# train benigh strings to get bbigram
bbigramPro = NLP_module.getBigramPro(trainGDom)
        
# get malicious letters distribution
#file = open('/Users/lixiaodan/Desktop/ece590/DGA/conficker.txt', 'r')
file = open('/Users/lixiaodan/NLPkit/DataSet/DGA_data_set/zeus.txt', 'r')

dgas = list()
for line in file:
    dga = line.split('.')[0]
    dgas.append(dga)
    
mtotal = 0
trainDgas = dgas[0:80000]
testDgas = dgas[80000:]

munigramPro = NLP_module.getUnigramPro(trainDgas)

# train malicious strings to get mbigram
mbigramPro = NLP_module.getBigramPro(trainDgas)

# 1 - dga, 0 - valid domain
# testing benign dataset
gDomByUni = list()
gDomByBi = list()
for gdDom in testGDom:
    KLdisUni = NLP_module.getUniKLdis(bunigramPro, munigramPro, gdDom)
    if KLdisUni <= 0:
        gDomByUni.append(1)
    else :
        gDomByUni.append(0)

    KLdisBi = NLP_module.getBiKLdis(bbigramPro, mbigramPro, gdDom)
    if KLdisBi <= 0:
        gDomByBi.append(1)
    else:
        gDomByBi.append(0)

dgaByUni = list()
dgaByBi = list()
for dga in testDgas: 
    KLdisUni = NLP_module.getUniKLdis(bbigramPro, mbigramPro, dga)
    if KLdisUni <= 0:
        dgaByUni.append(1)
    else :
        dgaByUni.append(0)

    KLdisBi = NLP_module.getBiKLdis(bbigramPro, mbigramPro, dga)
    if KLdisBi <= 0:
        dgaByBi.append(1)
    else:
        dgaByBi.append(0)

y_label = [0] * 20000 + [1] * 20000
resultByUni = gDomByUni + dgaByUni
resultByBi = gDomByBi + dgaByBi

NLP_module.plotRoc(resultByUni, y_label)
NLP_module.plotRoc(resultByBi, y_label)