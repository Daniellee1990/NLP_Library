#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:08:29 2017

@author: lixiaodan
"""

import numpy as np
import math
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# get the K-L divergence for unigram
def getUniKLdis(bunigram, munigram, inputStr):
    KL_dis = 0
    N = len(inputStr)
    pxg = 1
    for c in inputStr:
        if c in bunigram.keys() and bunigram.get(c) != None:
            pxg = pxg * bunigram.get(c)
        else:
            pxg = pxg * 10e-6
    
    # get P(x|b)
    pxb = 1
    for c in inputStr:
        if c in munigram.keys() and munigram.get(c) != None:
            pxb = pxb * munigram.get(c)
        else:
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
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_aucUni)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def getUnigramPro(dataset):
    bunigram = dict()
    totalCh = 0
    for domain in dataset:
        totalCh = totalCh + len(domain)
        for char in domain:
            if char in bunigram.keys():
                bunigram[char] = bunigram.get(char) + 1
            else:
                bunigram[char] = 1
    for key in bunigram.keys():
        bunigram[key] = bunigram[key] * 1.0 / totalCh
    return bunigram

def getUnigram(dataset):
    bunigram = dict()
    for domain in dataset:
        for char in domain:
            if char in bunigram.keys():
                bunigram[char] = bunigram.get(char) + 1
            else:
                bunigram[char] = 1
    return bunigram

def getBigramPro(dataset):
    bbigram = dict()
    tt = 0
    n = 2
    for domain in dataset:
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
    return bbigram

def getBigram(dataset):
    bbigram = dict()
    n = 2
    for domain in dataset:
        if len(domain) <= 1:
            continue
        tp = ''
        for i in range(len(domain)):
            tp = tp + domain[i]
            if i >= (n - 1):
                if tp in bbigram.keys():
                    bbigram[tp] = bbigram.get(tp) + 1
                else :
                    bbigram[tp] = 1
                tp = tp[1:]
    return bbigram

def getBiList(dataset):
    biList = dict()
    tt = 0
    n = 2
    for domain in dataset:
        if len(domain) <= 1:
            continue
        tt = tt + len(domain) - (n - 1)
        tp = ''
        curBis = list()
        for i in range(len(domain)):
            tp = tp + domain[i]
            if i >= (n - 1):
                curBis.append(tp)    
                tp = tp[1:]
        biList[domain] = curBis
    return biList

def getPPWs(testSet, logMatrix, rowNames):
    PPWs = list()
    for test in testSet:
        testStr = '^'+ test + '$'
        N = len(testStr)
        PPW = 1
        cur = ''
        for i in range(N):
            cur = cur + testStr[i]
            if i >= 1:
                row = rowNames.index(cur[0])
                col = rowNames.index(cur[1])
                PPW = PPW + logMatrix[row][col]
        PPW = PPW * (-1.0) / N
        PPWs.append(PPW)
    return PPWs

def printStaticsPPW(url, biMatrix, rowNames):
    file = open(url, 'r')
    PPWs = list()
    for line in file:
        tp = line.split('.')[0]
        PPWs.append(tp)
    PPW = getPPWs(PPWs, biMatrix, rowNames)
    print(np.mean(PPW))
    print(np.min(PPW))
    print(np.max(PPW))
    
def getBiProMatrix(bunigram, chars, gDomains):
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
    # add v smoothing          
    for i in range(len(chars)):
        cwn_1 = bunigram.get(chars[i])
        for j in range(len(chars)):
            biMatrix[i][j] = ( biMatrix[i][j] + 1 ) * 1.0 / (cwn_1 + len(chars))
    return biMatrix

def minEditDisCost(src, des):
    if len(src) == 0:
        return len(des)
    if len(des) == 0:
        return len(src)
    dis = np.zeros((len(src) + 1, len(des) + 1))
    for i in range(len(src) + 1):
        dis[i][0] = i
    for j in range(len(des) + 1):
        dis[0][j] = j
    for i in range(1, len(src) + 1):
        for j in range(1, len(des) + 1):
            if src[i - 1] == des[j - 1]:
                dis[i][j] = dis[i - 1][j - 1]
            else:
                dis[i][j] = min(min(dis[i - 1][j], dis[i][j - 1]) + 1, dis[i - 1][j - 1] + 2)
    return dis[len(src)][len(des)]

def minEditDis(src, des):
    if len(src) == 0:
        return len(des)
    if len(des) == 0:
        return len(src)
    dis = np.zeros((len(src) + 1, len(des) + 1))
    for i in range(len(src) + 1):
        dis[i][0] = i
    for j in range(len(des) + 1):
        dis[0][j] = j
    for i in range(1, len(src) + 1):
        for j in range(1, len(des) + 1):
            if src[i - 1] == des[j - 1]:
                dis[i][j] = dis[i - 1][j - 1]
            else:
                dis[i][j] = min(min(dis[i - 1][j], dis[i][j - 1]) + 1, dis[i - 1][j - 1] + 1)
    return dis[len(src)][len(des)]

def minEditDisWithSwap(src, des):
    if len(src) == 0:
        return len(des)
    if len(des) == 0:
        return len(src)
    dis = np.zeros((len(src) + 1, len(des) + 1))
    for i in range(len(src) + 1):
        dis[i][0] = i
    for j in range(len(des) + 1):
        dis[0][j] = j
    for i in range(1, len(src) + 1):
        for j in range(1, len(des) + 1):
            if src[i - 1] == des[j - 1]:
                dis[i][j] = dis[i - 1][j - 1]
            elif src[i - 1] == des[j - 2] and src[i - 2] == des[j - 1]:
                dis[i][j] = min(min(dis[i - 1][j], dis[i][j - 1]) + 1, dis[i - 1][j - 1] + 1)
                dis[i][j] = min(dis[i][j], dis[i - 2][j - 2] + 1)
            else:
                dis[i][j] = min(min(dis[i - 1][j], dis[i][j - 1]) + 1, dis[i - 1][j - 1] + 1)
    return dis[len(src)][len(des)]

def strCmp(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    if len1 != len2:
        return False
    str1 = str1.lower()
    str2 = str2.lower()
    for i in range(len1):
        if str1[i] != str2[i]:
            return False
    return True