#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:45:45 2017

@author: lixiaodan
"""
from nltk.corpus import wordnet as wn

# Thesaurus methods
# path-length based similarity:
dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')
similarity = dog.path_similarity(cat)
print(similarity)