#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:42:37 2017

@author: lixiaodan
"""

import NLP_module

str1 = "daniel"
str2 = "dadafasdfasdfasdfsa"
print(NLP_module.minEditDisCost(str1, str2))

str1 = "daniel"
str2 = "danopt"
print(NLP_module.minEditDisCost(str1, str2))

str1 = "daniel_dan"
str2 = "d2n"
print(NLP_module.minEditDis(str1, str2))
print(NLP_module.minEditDisCost(str1, str2))

str1 = "warren"
str2 = "warrant"
print("warren -> warrant")
print(NLP_module.minEditDis(str1, str2))

str1 = "phiosophy"
str2 = "philanthropy"
print("phiosophy -> philanthropy")
print(NLP_module.minEditDisWithSwap(str1, str2))

str1 = "phiosophy"
str2 = "phiasopyh"
print("phiosophy -> phiasopyh")
print(NLP_module.minEditDisWithSwap(str1, str2))

