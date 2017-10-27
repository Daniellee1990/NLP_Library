#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:48:18 2017

@author: lixiaodan
"""

import nltk
from nltk.corpus import treebank

#################### using seed words and adjective coordination ################
wsjs = nltk.corpus.treebank.fileids()
for fileName in wsjs:
    for cursent in treebank.tagged_sents(fileName):
        print(cursent)
        #for word in cursent:
            #print(word)