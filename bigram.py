#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:34:21 2017

@author: lixiaodan
"""

import pandas as pd
import numpy as np
import math

def getPPWs(testSet, biMatrix, chars):
    PPWs = list()
    for dga in testSet:
        testStr = '^'+ dga + '$'
        N = len(testStr)
        PPW = 1
        cur = ''
        for i in range(N):
            cur = cur + testStr[i]
            if i >= 1:
                row = chars.index(cur[0])
                col = chars.index(cur[1])
                PPW = PPW + logMatrix[row][col]
        PPW = PPW * (-1.0) / N
        PPWs.append(PPW)
    return PPWs

def printStatiscPPW(url):
    file = open(url, 'r')
    PPWs = list()
    for line in file:
        tp = line.split('.')[0]
        PPWs.append(tp)
    PPW = getPPWs(PPWs, biMatrix, chars)
    print(np.mean(PPW))
    print(np.min(PPW))
    print(np.max(PPW))

df = pd.read_csv('/Users/lixiaodan/Desktop/ece590/DGA/top-1m.csv', header=None)
goodNames = df[1]
gDomains = list()
for i in range(len(goodNames)):
    domainTp = goodNames[i].split('.')[0]
    gDomains.append('^'+ domainTp + '$')
trainGDom = gDomains[0:980000]
testGDom = gDomains[980000:]

bunigram = dict()
totalCh = 0
for domain in gDomains:
    totalCh = totalCh + len(domain)
    for char in domain:
        if char in bunigram.keys():
            bunigram[char] = bunigram.get(char) + 1
        else:
            bunigram[char] = 1
chars = list(bunigram.keys())
biMatrix = np.zeros((len(chars), len(chars)))

for domain in gDomains:
    cur = ''
    for i in range(len(domain)):
        cur = cur + domain[i]
        if i >=1:
           row = chars.index(cur[0])
           col = chars.index(cur[1])
           biMatrix[row][col] = biMatrix[row][col] + 1
           cur = cur[1:]
           
for i in range(len(chars)):
    cwn_1 = bunigram.get(chars[i])
    for j in range(len(chars)):
        biMatrix[i][j] = ( biMatrix[i][j] + 1 ) * 1.0 / (cwn_1 + len(chars))

logMatrix = np.zeros((len(chars), len(chars)))
for i in range(len(chars)):
    for j in range(len(chars)):
        logMatrix[i][j] = math.log2(biMatrix[i][j])

gPPWs = getPPWs(trainGDom, biMatrix, chars)
goodMean = np.mean(gPPWs)
print("PPW for valid domain name")
print(goodMean)
print(np.min(gPPWs))
print(np.max(gPPWs))

url = '/Users/lixiaodan/Desktop/ece590/DGA/conficker.txt'
print("PPW for conficker")
printStatiscPPW(url)

url = '/Users/lixiaodan/Desktop/ece590/DGA/cryptolocker.txt'
print("PPW for cryptolocker")
printStatiscPPW(url)

url = '/Users/lixiaodan/Desktop/ece590/DGA/opendns-random-domains.txt'
print("PPW for opendns-random-domains")
printStatiscPPW(url)

url = '/Users/lixiaodan/Desktop/ece590/DGA/zeus.txt'
print("PPW for zeus")
printStatiscPPW(url)

url = '/Users/lixiaodan/Desktop/ece590/DGA/tinba.txt'
print("PPW for tinba")
printStatiscPPW(url)

url = '/Users/lixiaodan/Desktop/ece590/DGA/rovnix.txt'
print("PPW for rovnix")
printStatiscPPW(url)

url = '/Users/lixiaodan/Desktop/ece590/DGA/ramdo.txt'
print("PPW for ramdo")
printStatiscPPW(url)

url = '/Users/lixiaodan/Desktop/ece590/DGA/pushdo.txt'
print("PPW for pushdo")
printStatiscPPW(url)
    



    

    



