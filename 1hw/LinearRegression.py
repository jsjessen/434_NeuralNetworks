#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 1
# Due 2019-09-05

import pandas as pd
import numpy as np

def linearRegression(target, predictors, dataPath, 
    emptyCellHandling = 'drop', 
    outputPrecision = 2):

    print('='*12 + ' Linear Regression ' + '='*12)
    print("Target: {}".format(target))
    predictorsString = 'Predictors: '
    for predictor in predictors:
        predictorsString += predictor + ' '
    print(predictorsString)

    columns = predictors.copy()
    columns.append(target)

    # Read data from file
    # Read CSV:  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    # DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    dataFrame = pd.read_csv(dataPath, usecols=columns)

    if emptyCellHandling.lower() == 'zero':
        print("Empty cells have been set to zero.")
        dataFrame = dataFrame.fillna(0)
    elif emptyCellHandling.lower() == 'drop':
        print("Rows with empty cells have been dropped.")
        dataFrame = dataFrame.dropna()
    else:
        print("Warning: Empty cells not handled.")

    numRows = len(dataFrame.index)
    numPredictors = len(predictors)

    # [[ 1 Sugars0 Fiber0 ]
    #  [ 1 Sugars1 Fiber1 ]
    #          ...
    #  [ 1 SugarsN FiberN ]]
    X = np.ones((numRows, 1 + numPredictors)) # First column ones for bias node
    X[:,1:] = dataFrame[predictors].values

    y = dataFrame[target].values

    # Solve the normal equation
    w = np.linalg.solve(X.T.dot(X), X.T.dot(y))

    print('-'*43)
    print("Bias = {}".format(round(w[0], outputPrecision)))
    for i, predictor in enumerate(predictors):
        print("{0} slope = {1}".format(predictor, round(w[1+i], outputPrecision)))

    yFit = X.dot(w)
    # residuals = yFit - y

    yAvg = np.mean(y)

    # Identity: SST = SSR + SSE
    # Sum of Squares Regression
    ssrDiff = yFit - yAvg
    SSR = np.sum(ssrDiff.dot(ssrDiff))
    # Sum of Squares Error
    sseDiff = y - yFit
    SSE = np.sum(sseDiff.dot(sseDiff))
    # Sum of Squares Total 
    sstDiff = y - yAvg
    SST = np.sum(sstDiff.dot(sstDiff))

    # Coefficient of Determination
    r2 = SSR / SST

    # Mean Squared Error
    MSE = SSE / (numRows - numPredictors - 1)

    # Standard Error of Estimation
    s = np.sqrt(MSE)

    print("R^2 = {}%".format(round(r2 * 100, outputPrecision)))
    print("s = {}".format(round(s, outputPrecision)))

    return r2, s