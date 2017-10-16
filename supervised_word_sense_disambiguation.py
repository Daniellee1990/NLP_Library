#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:39:59 2017

@author: lixiaodan
"""

from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
#from nltk.wsd import lesk

def strCmp(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    if len1 != len2:
        return False
    for i in range(len1):
        if str1[i] != str2[i]:
            return False
    return True

# get the WSD by simple_lesk
def SimpleLesk(word, sent):
    res = dict()
    res["word"] = word
    res["base"] = wn.morphy(word)
    res["synsets"] = wn.synsets(word)
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sent)
    
    best_sense = res["synsets"][0]
    max_overlap = 0
    
    stop_wds = stopwords.words("english")
    stop_wds.extend([res["word"], res["base"]])
    
    context = list()
    for i in range(len(tokens)):
        if tokens[i] not in stop_wds:
            context.append(tokens[i].lower())
    context = set(context)
    ## get sense
    for sense in res["synsets"]:
        signature = set()
        text = list()
        for ex in sense.examples():
            text.append(ex)
        text.append(sense.definition())
        total = " ".join(text)
        words = tokenizer.tokenize(total)
        for word in words:
            if word not in stop_wds:
                signature.add(word.lower())
        curOL = len(signature.intersection(context))
        if curOL > max_overlap:
            max_overlap = curOL
            best_sense = sense
    return best_sense
    
sent = "The bank can guarantee deposits will eventually cover future tuition costs because it invests in adjustable-rate mortgage securities."
word = 'bank'

sense = SimpleLesk(word, sent)
print(sense.definition())

window_len = 2

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(sent)
tags = pos_tag(tokens)

word_pos = -1
for i in range(len(tags)):
    if strCmp(tags[i][0], word):
        word_pos = i
# get the collocational feature vector
col_feature = list()
for k in range(word_pos - window_len, word_pos):
    col_feature.append(tags[k][0])
    col_feature.append(tags[k][1])

for k in range(word_pos + 1, word_pos + 3):
    col_feature.append(tags[k][0])
    col_feature.append(tags[k][1])

before_pair = ''
for j in range(word_pos - 1, word_pos - window_len - 1, -1):
    before_pair = before_pair + " " + tokens[j]
col_feature.append(before_pair)

after_pair = ''    
for i in range(word_pos + 1, word_pos + window_len + 1):
    after_pair = after_pair + " " + tokens[i]
col_feature.append(after_pair)  

