#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 1
# Due 2019-09-05

# Python Tutorial:      https://docs.python.org/3.7/tutorial/index.html
# Python Documentation: https://docs.python.org/3.7/index.html

# python 1hw/hw1.py > 1hw/output.txt

from LinearRegression import linearRegression as linReg

dataPath = 'data/Cereals.csv'
target = 'Rating'
precision = 2
emptyMethods = ['zero', 'drop']

# Write a code for regression of nutritional rating vs sugar and fiber. 
# Train with example from Cereals dataset on class web page
# Report bias and slopes for predictors sugar and fiber,
# coefficient of determination, R2, and standard error of estimation, s.

# Consider protein, fat, and sodium separately as a third attribute, 
# in addition to sugar and fiber, to predict the nutritional rating cereals.
# Report the change in R2 and s relative to sugar and fiber only.

thirdAttributes = ['Protein', 'Fat', 'Sodium']

for emptyMethod in emptyMethods:
    predictors = ['Sugars', 'Fiber']
    baseR2, baseS = linReg(target, predictors, dataPath, 
        emptyCellHandling=emptyMethod, outputPrecision=precision)
    print("R² = {}%".format(round(baseR2 * 100, precision)))
    print("s = {}".format(round(baseS, precision)))
    print('_'*43 + '\n')
    
    for thirdAttribute in thirdAttributes:
        predictors = ['Sugars', 'Fiber']
        predictors.append(thirdAttribute)
        r2, s = linReg(target, predictors, dataPath, 
            emptyCellHandling=emptyMethod, outputPrecision=precision)
        print("R² = {0}% Δ({1:+}%)".format(round(r2 * 100, precision), round((r2 - baseR2)*100, precision)))
        print("s = {0} Δ({1:+})".format(round(s, precision), round(s - baseS, precision)))
        print('_'*43 + '\n')

    print('\n' + '#'*43 + '\n')