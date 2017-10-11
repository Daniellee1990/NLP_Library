#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:34:21 2017

@author: lixiaodan
"""

import pandas as pd
import numpy as np
import math
import NLP_module

df = pd.read_csv('/Users/lixiaodan/Desktop/ece590/DGA/top-1m.csv', header=None)
goodNames = df[1]
gDomains = list()
for i in range(len(goodNames)):
    domainTp = goodNames[i].split('.')[0]
    gDomains.append('^'+ domainTp + '$')
trainGDom = gDomains[0:980000]
testGDom = gDomains[980000:]

bunigram = NLP_module.getUnigram(gDomains)

chars = list(bunigram.keys())
biMatrix = NLP_module.getBiProMatrix(bunigram, chars, gDomains)

logMatrix = np.zeros((len(chars), len(chars)))
for i in range(len(chars)):
    for j in range(len(chars)):
        logMatrix[i][j] = math.log2(biMatrix[i][j])
        
gPPWs = NLP_module.getPPWs(trainGDom, logMatrix, chars)
goodMean = np.mean(gPPWs)
print("PPW for valid domain name")
print(goodMean)
print(np.min(gPPWs))
print(np.max(gPPWs))

url = '/Users/lixiaodan/Desktop/ece590/DGA/conficker.txt'
print("PPW for conficker")
NLP_module.printStaticsPPW(url, logMatrix, chars)

url = '/Users/lixiaodan/Desktop/ece590/DGA/cryptolocker.txt'
print("PPW for cryptolocker")
NLP_module.printStaticsPPW(url, logMatrix, chars)

url = '/Users/lixiaodan/Desktop/ece590/DGA/opendns-random-domains.txt'
print("PPW for opendns-random-domains")
NLP_module.printStaticsPPW(url, logMatrix, chars)

url = '/Users/lixiaodan/Desktop/ece590/DGA/zeus.txt'
print("PPW for zeus")
NLP_module.printStaticsPPW(url, logMatrix, chars)

url = '/Users/lixiaodan/Desktop/ece590/DGA/tinba.txt'
print("PPW for tinba")
NLP_module.printStaticsPPW(url, logMatrix, chars)

url = '/Users/lixiaodan/Desktop/ece590/DGA/rovnix.txt'
print("PPW for rovnix")
NLP_module.printStaticsPPW(url, logMatrix, chars)

url = '/Users/lixiaodan/Desktop/ece590/DGA/ramdo.txt'
print("PPW for ramdo")
NLP_module.printStaticsPPW(url, logMatrix, chars)

url = '/Users/lixiaodan/Desktop/ece590/DGA/pushdo.txt'
print("PPW for pushdo")
NLP_module.printStaticsPPW(url, logMatrix, chars)