#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:15:35 2017

@author: lixiaodan
"""

from gensim.models import Word2Vec
from nltk.corpus import brown
b = Word2Vec(brown.sents())
b.most_similar('man', topn=5)
print(b)