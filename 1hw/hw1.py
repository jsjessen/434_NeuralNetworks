#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 1
# Due 2019-09-05

# Python Tutorial:      https://docs.python.org/3.7/tutorial/index.html
# Python Documentation: https://docs.python.org/3.7/index.html

from LinearRegression import linearRegression as linReg

dataPath = 'data/Cereals.csv'
target = 'Rating'
precision = 2

# Write a code for regression of nutritional rating vs sugar and fiber. 
# Train with example from Cereals dataset on class web page
# Report bias and slopes for predictors sugar and fiber,
# coefficient of determination, R2, and standard error of estimation, s.

# Consider protein, fat, and sodium separately as a third attribute, 
# in addition to sugar and fiber, to predict the nutritional rating cereals.
# Report the change in R2 and s relative to sugar and fiber only.

predictors = ['Sugars']
r0, s0 = linReg(target, predictors, dataPath, emptyCellHandling='zero', outputPrecision=precision)

predictors = ['Sugars', 'Fiber']
r1, s1 = linReg(target, predictors, dataPath, emptyCellHandling='zero', outputPrecision=precision)

print("R2:")
print(r1 - r0)
print("s:")
print(s1 - s0)