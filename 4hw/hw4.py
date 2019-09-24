#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 4
# Due 2019-10-01

# Python Tutorial:      https://docs.python.org/3.7/tutorial/index.html
# Python Documentation: https://docs.python.org/3.7/index.html

# python 4hw/hw4.py > 4hw/output.txt

from PerceptronClassification import perceptronClassification as pc

dataPath = 'data/glassDataWithHeader.csv'
inputColumns = list(range(0,9))
classColumn = 9
precision = 2
emptyMethods = ['zero', 'drop']

rSq, s = pc(dataPath, inputColumns, classColumn, 
    normalizeInputs=False,
    emptyCellHandling='drop', 
    outputPrecision=precision)

# print('\n')
# print("R² = {}%".format(round(rSq * 100, precision)))
# print("s = {}".format(round(s, precision)))