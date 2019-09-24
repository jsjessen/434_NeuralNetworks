#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 4
# Due 2019-10-01

# Python Tutorial:      https://docs.python.org/3.7/tutorial/index.html
# Python Documentation: https://docs.python.org/3.7/index.html

# python 4hw/hw4.py > 4hw/output.txt

import pandas as pd
import numpy as np
from PerceptronClassification import perceptronClassification as pc

dataPath = 'data/glassDataWithHeader.csv'
inputColumns = list(range(0,9))
classColumn = 9
precision = 2
totalLabel = 'Total'
accuracyLabel = 'Accuracy'

classifications, fits, rSq = pc(dataPath, inputColumns, classColumn)

classLabels = np.unique(classifications)
classLabels = np.sort(classLabels)

totals = pd.DataFrame(0, index=[totalLabel, accuracyLabel], columns=classLabels)
confusionMatrix = pd.DataFrame(0, index=classLabels, columns=classLabels)
for classLabel, fit in zip(classifications, fits):
    totals.loc[totalLabel, classLabel] += 1
    if fit < 1.5: assignedClass = 1
    elif fit > 4: assignedClass = 6
    else: assignedClass = 2
    confusionMatrix.loc[assignedClass, classLabel] += 1

totalCorrectlyClassified = 0
totalClassified = 0
for classLabel in classLabels:
    numCorrectlyClassified = confusionMatrix.loc[classLabel, classLabel]
    numClassified = totals.loc[totalLabel, classLabel]
    accuracyPercentage = (numCorrectlyClassified / numClassified) * 100
    totals.loc[accuracyLabel, classLabel] = str(round(accuracyPercentage, precision)) + '%'
    totalCorrectlyClassified += numCorrectlyClassified
    totalClassified += numClassified

# Report the following:
#     coefficient of determination, R^2
print("R² = {}%\n".format(round(rSq * 100, precision)))

#     number of records in each class and accuracy of class assignment
print(totals)

#     overall accuracy
overallAccuracy = (totalCorrectlyClassified / totalClassified) * 100
print("\nOverall Accuracy: {}%".format(round(overallAccuracy, precision)))

#     3-way confusion matrix as shown below
print("\nConfusion Matrix")
print('-'*20)
print(confusionMatrix)
print('-'*20)
print("Row: assigned class")
print("Col: actual class")

# in class 1 assigned class 1	# in class 2 assigned class 1 	 # in class 6 assigned class 1 
# in class 1 assigned class 2 	# in class 2 assigned class 2 	 # in class 6 assigned class 2 
# in class 1 assigned class 6	# in class 2 assigned class 6 	 # in class 6 assigned class 6