#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:16:00 2017

@author: lixiaodan
"""
import numpy as np
#import lda

xi = 100
alpha = np.ones((3,))
words = ['horse', 'bridle', 'fly', 'helium']
topics = ['horse', 'donkey', 'ballon']
beta = np.array([[0.7, 0.2, 0.1, 0], [0.5, 0.5, 0, 0], [0.1, 0, 0.3, 0.6]])

#for w in D
# for one document
N = np.random.poisson(xi)
theta = np.random.dirichlet(alpha)
z = list()
w = list()
for i in range(N):
     tp = np.random.multinomial(1, theta)
     curTopic = -1
     for i in range(len(tp)):
         if tp[i] != 0:
             curTopic = i
     z.append(topics[curTopic])
     curBeta = beta[curTopic, :]
     tp2 = np.random.multinomial(1, curBeta)
     curWord = ""
     for i in range(len(tp2)):
         if tp2[i] != 0:
             curWord = words[i]
     w.append(curWord)

 