#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 4
# Due 2019-10-01

import pandas as pd
import numpy as np

def perceptronClassification(dataPath, inputColumns, classColumn,
    normalizeInputs = False,
    emptyCellHandling = 'drop', 
    outputPrecision = 2):

    # print("Input Columns: {}".format(inputColumns))
    # print("Class Column: {}".format(classColumn))

    columns = inputColumns.copy()
    columns.append(classColumn)

    # Read data from file
    # Read CSV:  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    # DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    data = pd.read_csv(dataPath, usecols=columns)

    numInputs = len(inputColumns)
    numRows = len(data.index)

    if emptyCellHandling.lower() == 'zero':
        print("Empty cells have been set to zero.")
        data = data.fillna(0)
    elif emptyCellHandling.lower() == 'drop':
        numRowsBefore = numRows
        data = data.dropna()
        numRows = len(data.index)
        numRowsDropped = numRowsBefore - numRows
        if numRowsDropped > 0:
            print("{} rows with empty cells have been dropped.".format(numRowsDropped))
    else:
        print("Warning: Empty cells not handled.")

    # Normalize Data
    if normalizeInputs:
        # TODO: Prevent class column from being affected
        data = (data - data.min()) / (data.max() - data.min())
    else:
        print("Warning: Not normalizing inputs")

    V = np.ones((numRows, 1 + numInputs)) # First column ones for bias node
    V[:,1:] = data.iloc[:,inputColumns].values
    
    classifications = data.iloc[:,classColumn].values

    # Solve the normal equation
    w = np.linalg.solve(V.T.dot(V), V.T.dot(classifications))

    fit = V.dot(w)
    yAvg = np.mean(classifications) # AKA yBar

    # Sum of Squares Regression
    #   Measures variability of fit from mean response.
    ssrDiff = fit - yAvg
    SSR = ssrDiff.dot(ssrDiff)

    # Sum of Squares Error
    #   Measures variability of response from all other sources after the linear relationship 
    #   between response and attributes has been accounted for.
    residuals = classifications - fit # Deviations predicted from actual empirical values of data
    SSE = residuals.dot(residuals)

    # Sum of Squares Total 
    #   The sum of the squared differences of each observation from the overall mean.
    #   Identity: SST = SSR + SSE
    delY = classifications - yAvg
    SST = delY.dot(delY)

    # Coefficient of Determination
    #   Interpreted as the fraction of the total variation of response over 
    #   the dataset that is explained by the linear fit.
    rSq = SSR / SST
    # Always increases when an additional attribute is included.
    # To be useful, a new attribute must significantly increase R2.

    # Mean Squared Error
    MSE = SSE / (numRows - numInputs - 1)

    # Standard Error of Estimation
    #   Interpreted as the typical size of residuals 
    s = np.sqrt(MSE)
    # Can be lower or higher when another attribute is added to the model.

    classLabels = np.unique(classifications)
    classLabels = np.sort(classLabels)

    # TODO: Move to hw4.py
    totalLabel = 'Total'
    accuracyLabel = 'Accuracy'
    totals = pd.DataFrame(0, index=[totalLabel, accuracyLabel], columns=classLabels)
    confusionMatrix = pd.DataFrame(0, index=classLabels, columns=classLabels)
    for index in data.index:
        classLabel = classifications[index]
        totals.loc[totalLabel, classLabel] += 1
        if fit[index] < 1.5: assignedClass = 1
        elif fit[index] > 4: assignedClass = 6
        else: assignedClass = 2
        confusionMatrix.loc[assignedClass, classLabel] += 1
    
    totalCorrectlyClassified = 0
    superTotalClassified = 0
    for classLabel in classLabels:
        numCorrectlyClassified = confusionMatrix.loc[classLabel, classLabel]
        totalCorrectlyClassified += numCorrectlyClassified
        totalClassified = totals.loc[totalLabel, classLabel]
        superTotalClassified += totalClassified
        totals.loc[accuracyLabel, classLabel] = str(round((numCorrectlyClassified / totalClassified) * 100, outputPrecision)) + '%'

    # Report the following:
    #     coefficient of determination (AKA R^2)
    print("\nR² = {}%\n".format(round(rSq * 100, outputPrecision)))

    #     number of records in each class and accuracy of class assignment
    print(totals)
    
    #     overall accuracy
    print("\nOverall Accuracy: {}%".format(round(((totalCorrectlyClassified / superTotalClassified) * 100), outputPrecision)))

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
 
    return rSq, s