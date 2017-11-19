#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 21:18:28 2017

@author: lixiaodan
"""

import email
import os
import re
import html2text
from nltk.corpus import stopwords
import math
import NLP_module
import ast

"""
Method
Removing the following tokens from the vocabulary:
punctuations and numbers
english stopwords
one and two letter words
"""

def parse(words_table, strings):
    h = html2text.HTML2Text()
    txt = h.handle(strings)
    txt = txt.lower()
    p = re.compile('\W+')
    splits  = p.split(txt)
    for word in splits:
        found = re.search('[0-9]+', word)
        if found != None:
            continue
        stop_wds = stopwords.words("english")
        stop_wds.extend(["email", "www", "com", "http", "html", "gif"])
        if word in stop_wds:
            continue
        if len(word) <= 2:
            continue
        if word in words_table.keys():
            words_table[word] = words_table.get(word) + 1
        else:
            words_table[word] = 1

def processTable(words_table):
    total_size = 0
    dict_after_process = dict()
    for word in words_table.keys():
        total_size = total_size + words_table.get(word)
    for word in words_table.keys():
        found = re.search('__', word)
        if found != None:
            continue
        if words_table[word] >= 5:
            dict_after_process[word] = words_table[word]
    return dict_after_process
 
labels = dict()
label_path = '/Users/lixiaodan/Desktop/ece590/CSDMC2010_SPAM/CSDMC2010_SPAM/SPAMTrain.label'
infile = open(label_path,'r')
label_List = list()
for line in infile:
    tp = line.split(" ")[1]
    eml_name = tp.split("\n")[0]
    labels[eml_name] = line.split(" ")[0]
    label_List.append(line.split(" ")[0])
infile.close()

path = '/Users/lixiaodan/Desktop/ece590/CSDMC2010_SPAM/CSDMC2010_SPAM/training_new'
listing = os.listdir(path)

fail_IO = list()
words_table = dict()
gd_table = dict()
bad_table = dict()
gd_cnt = 0
bad_cnt = 0

for i in range(len(listing)):
    fle = listing[i]
    if str.lower(fle[-3:])=="eml":
        try:
            msg = email.message_from_file(open(path + '/' + fle))
            strs = msg.as_string()
            if labels[fle] == "1":
                gd_cnt = gd_cnt + 1
                parse(gd_table, strs)
                gd_table = processTable(gd_table)
            else:
                bad_cnt = bad_cnt + 1
                parse(bad_table, strs)
                bad_table = processTable(bad_table)
        except UnicodeDecodeError:
            fail_IO.append(fle)
            continue

# merge good and bad tables
good_keys = list(gd_table.keys())
bad_keys = list(bad_table.keys())            
all_words = good_keys + bad_keys
all_words = set(all_words)
all_words_list = list(all_words)
for i in range(len(all_words)):
    curWord = all_words_list[i]
    cnt1 = 0
    cnt2 = 0
    if curWord in gd_table.keys():
        cnt1 = gd_table.get(curWord)
    if curWord in bad_table.keys():
        cnt2 = bad_table.get(curWord)
    curcnt = cnt1 + cnt2
    words_table[curWord] = curcnt

# get prior probablity
total_email = gd_cnt + bad_cnt
pGood = gd_cnt * 1.0 / total_email
pBad = bad_cnt * 1.0 / total_email

# get likelihood for bad words
v = len(words_table.keys())
total_bad = 0
poster_bad = dict()
for bad_wd in bad_table.keys():
    total_bad = total_bad + bad_table.get(bad_wd)
for bad_wd in bad_table.keys():
    poster_bad[bad_wd] = (bad_table.get(bad_wd) + 1.0) / (total_bad + v)

# get likelihood for good words
total_gd = 0
poster_gd = dict()
for gd_wd in gd_table.keys():
    total_gd = total_gd + gd_table.get(gd_wd)
for gd_wd in gd_table.keys():
    poster_gd[gd_wd] = (gd_table.get(gd_wd) + 1.0) / (total_gd + v)
    
# The test case and get the posterior probability
test_path = '/Users/lixiaodan/Desktop/ece590/CSDMC2010_SPAM/CSDMC2010_SPAM/test_new'
tests = os.listdir(test_path)
result = list()
predict = list()
test_labels = list()

##1 stands for a HAM and 0 stands for a SPAM
for i in range(len(tests)):
    fle = tests[i]
    if str.lower(fle[-3:])=="eml":
        cur_table = dict()
        try:
            msg = email.message_from_file(open(test_path + '/' + fle))
            strs = msg.as_string()
            parse(cur_table, strs)
            cur_table = processTable(cur_table)
            proBad = 1
            proGd = 1
            ######## get the bad posterior probability for email
            for wd in cur_table.keys():
                if wd in poster_bad.keys():
                    proBad = proBad * math.pow(poster_bad[wd], cur_table[wd])
                else:
                    proBad = proBad * math.pow((1.0 / (total_bad + v)), cur_table[wd])
            proBad = proBad * pBad
            ######## get the good posterior probability for email
            for wd in cur_table.keys():
                if wd in poster_gd.keys():
                    proGd = proGd * math.pow(poster_gd[wd], cur_table[wd])
                else:
                    proGd = proGd * math.pow((1.0 / (total_gd + v)), cur_table[wd])
            proGd = proGd * pGood
            pair = list()
            pair.append(fle)
            if proBad > proGd:
                pair.append(0)
                predict.append(0)
            else:
                pair.append(1)
                predict.append(1)
            test_labels.append(ast.literal_eval(labels[fle]))
            result.append(pair)
        except UnicodeDecodeError:
            fail_IO.append(fle)
            continue
        
NLP_module.plotRoc(predict, test_labels)
detec_spam = 0
tt_spam = 0
detec_good = 0
tt_good = 0
for i in range(len(predict)):
    if test_labels[i] == 0:
        tt_spam = tt_spam + 1
        if predict[i] == 0:
            detec_spam = detec_spam + 1
    if test_labels[i] == 1:
        tt_good = tt_good + 1
        if predict[i] == 1:
            detec_good = detec_good + 1
spam_rate = 1.0 * detec_spam / tt_spam
good_rate = 1.0 * detec_good / tt_good 
accuracy = 1.0 * (detec_spam + detec_good) / (tt_spam + tt_good)      