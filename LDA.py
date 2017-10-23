#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:16:00 2017

@author: lixiaodan
"""
import numpy as np
import gensim.models
from gensim import corpora
from nltk.tokenize import RegexpTokenizer

xi = 100
alpha = np.ones((3,))
words = ['horse', 'bridle', 'fly', 'helium']
topics = ['horse', 'donkey', 'ballon']
beta = np.array([[0.7, 0.2, 0.1, 0], [0.5, 0.5, 0, 0], [0.1, 0, 0.3, 0.6]])
numDoc = 10
result = list()

for i in range(numDoc):
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
    result.append(w)
results = '\n'.join([' '.join(doc) for doc in result])

"""
dictionary = corpora.Dictionary(result)
corpus = [dictionary.doc2bow(text) for text in result]
print(corpus[0])
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=3, num_words=4))   
""" 

mainlist = list()
tokenizer = RegexpTokenizer(r'\w+')
infile = open('/Users/lixiaodan/Desktop/ece590/text.txt','r')
for line in infile:
    mainlist.append(line)
docs = list()
for line in mainlist:
    raw = line.lower()
    curWords = tokenizer.tokenize(raw)
    docs.append(curWords)
word2cnt = corpora.Dictionary(docs)
corpus = [word2cnt.doc2bow(line) for line in docs]
print(corpus[0])
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = word2cnt, passes=20)
print(ldamodel.print_topics(num_topics=3, num_words = len(word2cnt.keys())))   