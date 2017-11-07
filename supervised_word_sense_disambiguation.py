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
import nltk
from nltk.corpus import treebank
import re 
from operator import itemgetter
import NLP_module

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
    
sent = "President Donald Trump’s executive order on health care issued Thursday marks the first major salvo in what the White House promises will be an extensive, targeted campaign to unravel the Affordable Care Act administratively."
word = 'Care'
word = word.lower()

# simple lesk algorithm
sense = SimpleLesk(word, sent)
print(sense.definition())

# Extraction of features:
window_len = 2

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(sent)
tags = pos_tag(tokens)

word_pos = -1
for i in range(len(tags)):
    if NLP_module.strCmp(tags[i][0], word):
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

## get BoW (bag of wrod) features
# get the WSJ and parse the document
# sumarize the word and its frequency
wsjs = nltk.corpus.treebank.fileids()
word2num = dict()
occurence = 0
for fileName in wsjs:
    for cursent in treebank.sents(fileName):
        if word in cursent:
            occurence = occurence + 1
            for wd in cursent:
                pattern = '[a-z]+'
                found = re.search(pattern, wd)
                if re.search(pattern, wd) is None:
                    continue
                stop_wds = stopwords.words("english")
                stop_wds.extend([word, wn.morphy(word), "'s"])
                if wd.lower() in stop_wds:
                    continue
                if wd.lower() in word2num:
                    word2num[wd.lower()] = word2num.get(wd.lower()) + 1
                else:
                    word2num[wd.lower()] = 1
                    
if occurence == 0:
    print("The current word does not appear in WSJ database")
    
maxnum = 10
sorted_dict = sorted(word2num.items(), key=itemgetter(1) , reverse=True)
top_words = list()

if maxnum > len(sorted_dict):
    print("The bag of words size is more than that of neighboring words in WSJ")
    
for i in range(maxnum):
    top_words.append(sorted_dict[i][0])
    
# get the binary vector/ bag of words
BoW = list()
for i in range(maxnum):
    BoW.append(0)
for wd in tokens:
    if wd.lower() in top_words:
        BoW[top_words.index(wd.lower())] = BoW[top_words.index(wd.lower())] + 1