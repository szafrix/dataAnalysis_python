# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:14:14 2020

@author: Szafran
"""

TP = 2
TN = 2
FP = 1
FN = 2

recall= TP/(TP+FN)
#Out of all the positive classes, how much we predicted correctly. It should be high as possible.

precision = TP/(TP+FP)
#Out of all the positive classes we have predicted correctly, how many are actually positive.

accuracy = (TP+TN)/(TP+TN+FP+FN)

F_score = 2*recall*precision/(recall+precision)