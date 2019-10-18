#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 8
# Due 2019-10-29

# Python Tutorial:      https://docs.python.org/3.7/tutorial/index.html
# Python Documentation: https://docs.python.org/3.7/index.html

# python 8hw/hw8.py > 8hw/output.txt

import pandas as pd
import numpy as np

dataPath = '8hw/predictions.csv' # actual | predicted
precision = 2
totalLabel = 'Total'
accuracyLabel = 'Accuracy'

# classifications, fits, rSq = pc(dataPath, inputColumns, classColumn)

# Read data from file
# Read CSV:  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
# DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
data = pd.read_csv(dataPath)

actualClasses = data.loc[:,'actual']
predictedClasses = data.loc[:,'predicted']

classLabels = np.unique(actualClasses)
classLabels = np.sort(classLabels)

# Choose a bin boundary equal to the average of the labels. 
binBoundary = np.mean(classLabels)

totals = pd.DataFrame(0, index=[totalLabel, accuracyLabel], columns=classLabels)
confusionMatrix = pd.DataFrame(0, index=classLabels, columns=classLabels)
for actualClass, predictedClass in zip(actualClasses, predictedClasses):
    totals.loc[totalLabel, actualClass] += 1
    if predictedClass < binBoundary: 
        assignedClass = 1
    else: 
        assignedClass = 5
    confusionMatrix.loc[assignedClass, actualClass] += 1

totalCorrectlyClassified = 0
totalClassified = 0
for actualClass in classLabels:
    numCorrectlyClassified = confusionMatrix.loc[actualClass, actualClass]
    numClassified = totals.loc[totalLabel, actualClass]
    accuracyPercentage = (numCorrectlyClassified / numClassified) * 100
    totals.loc[accuracyLabel, actualClass] = str(round(accuracyPercentage, precision)) + '%'
    totalCorrectlyClassified += numCorrectlyClassified
    totalClassified += numClassified


# number of records in each class and accuracy of class assignment
print(totals)

# Overall accuracy
overallAccuracy = (totalCorrectlyClassified / totalClassified) * 100
print("\nOverall Accuracy: {}%".format(round(overallAccuracy, precision)))

# Confusion matrix
matrixLabel = 'Confusion Matrix'
matrixLabelLength = len(matrixLabel)
print('\n' + matrixLabel)
print('-'*matrixLabelLength)
print(confusionMatrix)
print('-'*matrixLabelLength)
print("Row: assigned class")
print("Col: actual class")

# in class 1 assigned class 1	# in class 5 assigned class 1
# in class 1 assigned class 5 	# in class 5 assigned class 5