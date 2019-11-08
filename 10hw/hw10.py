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

# (Do not include the 10th column, which is class labels)
inputColumns = list(range(0,9))

# Calculate principal components of attributes in glass data short.csv
plotData = dr(dataPath, inputColumns)

# Report the eigenvalues ranked by decreasing magnitude.
# plotData.to_csv(r'10hw\output.csv', index=False, header=True)
print(plotData)

width = 0.5

fig, ax1 = plt.subplots()
plt.title('Proportion of Variance')

color = 'tab:blue'
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Variance Explained (%)', color=color)
ax1.set_ylim(0, 100)
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Variance Captured (%)', color=color)
ax2.set_ylim(0, 100)
ax2.tick_params(axis='y', labelcolor=color)

plotData['Variance Explained'].plot(ax=ax1, style='r-', kind='bar', width=width)
plotData['Proportion of Variance'].plot(ax=ax2, style='r-', secondary_y='Variance Captured (%)')
ax1.set_xticks(plotData.index)
ax1.set_xticklabels(plotData['Principal Component'], rotation=0)

plt.xlim(-width, len(plotData.index) - width)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()