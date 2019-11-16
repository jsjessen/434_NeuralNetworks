#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 11
# Due 2019-11-19

# Python Tutorial:      https://docs.python.org/3.7/tutorial/index.html
# Python Documentation: https://docs.python.org/3.7/index.html

# python 11hw/hw11.py > 11hw/output.txt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from OnlineMLP import onlineMLP as mlp

#      XOR
#   =========
#   x1	x2	r
#   ---------
#   0	0	0
#   0	1	1
#   1	0	1
#   1	1	0
def xor(a, b):
    if (a != 0 and a != 1) or (b != 0 and b != 1):
        raise ValueError("XOR inputs must be 0 or 1")
    return bool(a) != bool(b)

# Populate XOR dataset
xorDataset = []
falseTrue = [0, 1]
for x1 in falseTrue:
    for x2 in falseTrue:
        xorDataset.append([x1, x2, int(xor(x1, x2))])

# Hidden layer contains 3 nodes
# AKA: z
# numHiddenNodes = 3

# AKA: w
initial_inputWeights = [[-0.5,  1, -1],
                        [-0.5, -1,  1]]
# AKA: v
initial_hiddenWeights = [24, -20, -20]

# AKA: η, eta, etha
learningRate = 0.001

# No momentum parameter
# AKA: α, alpha

# Perform 1000 iterations
numIterations = 1000

Error, Z, decisionBoundary = mlp(
    xorDataset,
    initial_inputWeights, 
    initial_hiddenWeights, 
    learningRate, 
    numIterations)

# Create semilog plot of convergence.
plt.semilogy(Error)
plt.title('Convergence')
plt.xlabel('Iteration')
plt.ylabel('Error (SSR)')
plt.show()

# Report the final weight vectors and predicted y value for each example.


# With the final weights, calculate and report the values 
# z1 and z2 for each example in the dataset.
# 
# Use the values z1 and z2 associated with examples (0,0) and
# (0,1) to calculate the bias of a decision boundary with equal
# margins for the 2 classes. Report the margins and include a
# plot of feature space with the decision boundary and location of
# features associated with examples in the dataset.
plt.scatter(Z['z2'], Z['z1'])
z1 = [0, decisionBoundary]
z2 = [decisionBoundary, 0]
plt.plot(z1, z2, '--r', label='z1 + z2 = {}\nmargin = {}'.format(round(decisionBoundary, 3), 0.019))

plotMargin = 0.05

xlim = [np.min(Z['z1']), np.max(Z['z1'])]
xMargin = np.mean(xlim) * plotMargin
xlim[0] -= xMargin
xlim[1] += xMargin
plt.xlim(xlim)

ylim = [np.min(Z['z2']), np.max(Z['z2'])]
yMargin = np.mean(ylim) * plotMargin
ylim[0] -= yMargin
ylim[1] += yMargin
plt.ylim(ylim)

plt.title('XOR Feature Space with Decision Boundary')
plt.xlabel('z1')
plt.ylabel('z2')
plt.legend()
plt.show()