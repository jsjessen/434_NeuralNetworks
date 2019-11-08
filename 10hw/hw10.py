#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 10
# Due 2019-11-12

# Python Tutorial:      https://docs.python.org/3.7/tutorial/index.html
# Python Documentation: https://docs.python.org/3.7/index.html

# python 10hw/hw10.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DimensionalityReduction import dimensionalityReduction as dr

dataPath = 'data/glassDataWithHeader.csv'
inputColumns = list(range(0,9))

    # Calculate principal components of attributes in glass data short.csv
    # (Do not include the 10th column, which is class labels)

    # Report the eigenvalues ranked by decreasing magnitude. 

    # Calculate PoV for all eigenvalues and plot.

eigenvalues, proportionVariance = dr(dataPath, inputColumns)

eigValLabel = 'Ranked Eigenvalues'
print(eigValLabel)
print('-'*len(eigValLabel))
sumEigVals = np.sum(eigenvalues)
for i, eigVal in enumerate(eigenvalues):
    percent = (eigVal / sumEigVals) * 100
    print('{}) {} ({}%)'.format(i+1, round(eigVal, 4), round(percent, 2)))

# Recreate slide 14 plot
d = np.size(proportionVariance)
y_pos = np.arange(d)
plt.bar(y_pos, proportionVariance, align='center', alpha=0.5)
plt.xticks(y_pos, range(1, d+1))
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.title('Proportion of Variance')
plt.show()

# m1_t[['abnormal','fix','normal']].plot(kind='bar', width = width)
# m1_t['bad_rate'].plot(secondary_y=True)

# ax = plt.gca()
# plt.xlim([-width, len(m1_t['normal'])-width])
# ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'))