#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 1
# Due 2019-09-05

import pandas as pd
import numpy as np

datafile = 'data/Cereals.csv'

target = 'Rating'

predictors = ['Sugars','Fiber']
# predictors.append('Protein')
# predictors.append('Fat')
# predictors.append('Sodium')

attributes = predictors.copy()
attributes.append(target)

#==============================================================================

# Read data from file
df = pd.read_csv(datafile, usecols=attributes)
df = df.fillna(0) # Use 0 for empty cells
numRows = len(df.index)
numPredictors = len(predictors)

V = np.ones((numRows, 1 + numPredictors)) # First column ones for bias node
V[:,1:] = df[predictors].values

y = df[target].values

# Sove the normal equation
A = V.T @ V
b = V.T @ y
w = np.linalg.lstsq(A, b, rcond=None)[0] # Solves Aw = b
b0 = w[0]
bs = w[1]
bf = w[2]

print("Bias = {}".format(w[0]))
for i, predictor in enumerate(predictors):
    print("{0} slope = {1}".format(predictor, w[i]))

# Write a code for regression of nutritional rating vs sugar and fiber. 
# Train with example from Cereals dataset on class web page.
# Report bias and slopes for predictors sugar and fiber,
# coefficient of determination, R2, and standard error of estimation, s.




# Consider protein, fat, and sodium separately as a third attribute, 
# in addition to sugar and fiber, to predict the nutritional rating cereals.
# Report the change in R2 and s relative to sugar and fiber only.