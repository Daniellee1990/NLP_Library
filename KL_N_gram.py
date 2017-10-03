# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# get the K-L divergence for unigram
def getUniKLdis(bunigram, munigram, inputStr):
    KL_dis = 0
    # get P(x|g)
    N = len(inputStr)
    pxg = 1
    for c in inputStr:
        if c in bunigram.keys() and bunigram.get(tp) != None:
            pxg = pxg * bunigram.get(c)
        else: 
            #KL_dis = -1
            #break
            pxg = pxg * 10e-6
    
    # get P(x|b)
    pxb = 1
    for c in inputStr:
        if c in munigram.keys() and munigram.get(tp) != None:
            pxb = pxb * munigram.get(c)
        else:
            #KL_dis = 1
            #break
            pxb = pxb * 10e-6
    
    if KL_dis == 0:
        KL_dis = 1.0 / N * math.log(pxg * 1.0 / pxb)
    return KL_dis

# get Kullback-Liebler distance based on bigram
def getBiKLdis(bbigram, mbigram, inputStr):
    KL = 0
    pxg = 1
    N = len(inputStr)
    # get pxg
    tp = ""
    for i in range(len(inputStr)):
        tp = tp + inputStr[i]
        if i >= 1:
            if tp in bbigram.keys() and bbigram.get(tp) != None:
                pxg = pxg * bbigram.get(tp)
            else:
                #KL = -1 
                #break
                pxg = pxg * 10e-6
            tp = tp[1:]
            
    # get pxb
    pxb = 1
    tp = ""
    for i in range(len(inputStr)):
        tp = tp + inputStr[i]
        if i >= 1:
            if tp in bbigram.keys() and mbigram.get(tp) != None:
                pxb = pxb * mbigram.get(tp)
            else:
                #KL = 1 
                #break
                pxb = pxb * 10e-6
            tp = tp[1:]
    
    if KL == 0:
            KL = 1.0 / N * math.log(pxg * 1.0 / pxb)
    return KL

def plotRoc(y_predict, y_label):
    # roc for unigram
    fprUni, tprUni, _ = roc_curve(y_predict, y_label)
    roc_aucUni = auc(fprUni, tprUni)
    
    plt.figure()
    lw = 2
    plt.plot(fprUni, tprUni, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_aucUni)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of unigram')
    plt.legend(loc="lower right")
    plt.show()

df = pd.read_csv('/Users/lixiaodan/Desktop/ece590/DGA/top-1m.csv', header=None)
goodNames = df[1]
gDomains = list()
for i in range(len(goodNames)):
    domainTp = goodNames[i].split('.')[0]
    gDomains.append(domainTp)

trainGDom = gDomains[0:980000]
testGDom = gDomains[980000:]

# train benigh letters distibution to get bunigram 
bunigram = dict()
totalCh = 0
for domain in trainGDom:
    totalCh = totalCh + len(domain)
    for char in domain:
        if char in bunigram.keys():
            bunigram[char] = bunigram.get(char) + 1
        else:
            bunigram[char] = 1
for key in bunigram.keys():
    bunigram[key] = bunigram[key] * 1.0 / totalCh
    
# train benigh strings to get bbigram
bbigram = dict()
tt = 0
n = 2
for domain in trainGDom:
    if len(domain) <= 1:
        continue
    tt = tt + len(domain) - (n - 1)
    tp = ''
    for i in range(len(domain)):
        tp = tp + domain[i]
        if i >= (n - 1):
            if tp in bbigram.keys():
                bbigram[tp] = bbigram.get(tp) + 1
            else :
                bbigram[tp] = 1
            tp = tp[1:]
for key in bbigram.keys():
    bbigram[key] = bbigram[key] * 1.0 / tt
        
# get malicious letters distribution
file = open('/Users/lixiaodan/Desktop/ece590/DGA/conficker.txt', 'r')
dgas = list()
for line in file:
    dga = line.split('.')[0]
    dgas.append(dga)
    
munigram = dict()
mtotal = 0
trainDgas = dgas[0:80000]
testDgas = dgas[80000:]

for dga in trainDgas:
    mtotal = mtotal + len(dga)
    for c in dga:
        if c in munigram.keys():
            munigram[c] = munigram.get(c) + 1
        else:
            munigram[c] = 1
for key in munigram.keys():
    munigram[key] = munigram.get(key) * 1.0 / mtotal

# train malicious strings to get mbigram
mbigram = dict()
mtt = 0
n = 2
for dga in trainDgas:
    mtt = mtt + len(dga) - (n - 1)
    tp = ''
    for i in range(len(dga)):
        tp = tp + dga[i]
        if i >= (n - 1):
            if tp in mbigram.keys():
                mbigram[tp] = mbigram.get(tp) + 1
            else :
                mbigram[tp] = 1
            tp = tp[1:]
for key in mbigram.keys():
    mbigram[key] = mbigram.get(key) * 1.0 / mtt

# 1 - dga, 0 - valid domain
# testing benign dataset
gDomByUni = list()
gDomByBi = list()
for gdDom in testGDom:
    KLdisUni = getUniKLdis(bunigram, munigram, gdDom)
    if KLdisUni <= 0:
        gDomByUni.append(1)
    else :
        gDomByUni.append(0)

    KLdisBi = getBiKLdis(bbigram, mbigram, gdDom)
    if KLdisBi <= 0:
        gDomByBi.append(1)
    else:
        gDomByBi.append(0)

dgaByUni = list()
dgaByBi = list()
for dga in testDgas: 
    KLdisUni = getUniKLdis(bbigram, mbigram, dga)
    if KLdisUni <= 0:
        dgaByUni.append(1)
    else :
        dgaByUni.append(0)

    KLdisBi = getBiKLdis(bbigram, mbigram, dga)
    if KLdisBi <= 0:
        dgaByBi.append(1)
    else:
        dgaByBi.append(0)

y_label = [0] * 20000 + [1] * 20000
resultByUni = gDomByUni + dgaByUni
resultByBi = gDomByBi + dgaByBi


plotRoc(resultByUni, y_label)
plotRoc(resultByBi, y_label)

