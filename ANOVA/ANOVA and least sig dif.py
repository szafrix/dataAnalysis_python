# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:29:25 2020

@author: Szafran
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

os.chdir('C:\\Users\\Szafran\\Desktop\\python\\stat Fun\\ANOVA')

"""
Badanie polega na mierzeniu wagi swiń karmionych trzema rodzajami paszy
"""
np.random.seed(213413431)
pasza_A = np.random.randint(120, 150, 10)
pasza_B = np.random.randint(140, 170, 10)
pasza_C = np.random.randint(125, 140, 10)

data = pd.DataFrame(np.array([pasza_A, pasza_B, pasza_C]).T, columns=["pasza_A", 'pasza_B', 'pasza_C'])

print(data.describe())

#weryfikacja założenia o jednorodnosci wariancji: Test Bartletta
stats.bartlett(pasza_A, pasza_B, pasza_C)
#pvalue=0.16113920195400974, brak podstaw do odrzucenia hipotezy zerowej

plt.boxplot(data.T)
plt.savefig("boxplot")
plt.show()

"""
oczekujemy odrzucenia hipotezy zerowej na rzecz hipotezy alternatywnej
ze wzgledu na odstajaca srednia w grupie swin karmionych pasza B
"""

mean_Total = sum([data[x].sum() for x in data]) / sum([data[x].count() for x in data])
mean_groups = np.array([data[x].mean() for x in data])

ss_total = 0
for i in data:
    for j in data[i]:
        ss_total += (j-mean_Total)**2
        
ss_inside = 0
for i in data:
    for j in data[i]:
        ss_inside += (j-data[i].mean())**2
        
ss_between = 0
for i in data:
    ss_between += data[i].count()*(data[i].mean() - mean_Total)**2
    
f_stat = (ss_between/(2))/(ss_inside/(30-3))

stats.f.cdf(f_stat, 2, 27) #odrzucamy hipotezę zerową na rzecz hipoterzy alternatywnej

stats.f_oneway(pasza_A, pasza_B, pasza_C) #potwierdzenie

# porównanie wielokrotne
#procedura najmniejszej istotnej różnicy

lsd = stats.t.ppf(0.975, 27)*((ss_inside/(30-3))*0.2)**0.5
#as all samples have the same size, lsd is the same for all pairs

pair_1_2 = abs(pasza_A.mean() - pasza_B.mean())
pair_1_3 = abs(pasza_A.mean() - pasza_C.mean())
pair_2_3 = abs(pasza_B.mean() - pasza_C.mean())

# za zróżnicowanie odpowiadają pary 1 i 2 oraz 2 i 3, co oznacza, że 
# druga pasza odstaje od reszty.