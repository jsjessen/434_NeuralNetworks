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
    data = pd.read_csv(dataPath, usecols=columns)


    if emptyCellHandling.lower() == 'zero':
        print("Empty cells have been set to zero.")
        data = data.fillna(0)
    elif emptyCellHandling.lower() == 'drop':
        print("Rows with empty cells have been dropped.")
        data = data.dropna()
    else:
        print("Warning: Empty cells not handled.")

    numRows = len(data.index)
    numPredictors = len(predictors)

    # dataNorm = (data - data.min()) / (data.max() - data.min())

    # [[ 1 Sugars0 Fiber0 ]
    #  [ 1 Sugars1 Fiber1 ]
    #          ...
    #  [ 1 SugarsN FiberN ]]
    X = np.ones((numRows, 1 + numPredictors)) # First column ones for bias node
    X[:,1:] = data[predictors].values

    y = data[target].values

    # Solve the normal equation
    w = np.linalg.solve(X.T.dot(X), X.T.dot(y))

    print('-'*43)
    print("Bias = {}".format(round(w[0], outputPrecision)))
    for i, predictor in enumerate(predictors):
        print("{0} slope = {1}".format(predictor, round(w[1+i], outputPrecision)))

    yFit = X.dot(w)
    yAvg = np.mean(y)

    # Sum of Squares Regression
    #   Measures variability of fit from mean response.
    ssrDiff = yFit - yAvg
    SSR = np.sum(ssrDiff.dot(ssrDiff))

    # Sum of Squares Error
    #   Measures variability of response from all other sources after the linear relationship 
    #   between response and attributes has been accounted for.
    sseDiff = y - yFit # residuals
    SSE = np.sum(sseDiff.dot(sseDiff))

    # Sum of Squares Total 
    #   Identity: SST = SSR + SSE
    sstDiff = y - yAvg
    SST = np.sum(sstDiff.dot(sstDiff))

    # Coefficient of Determination
    #   Interpreted as the fraction of the total variation of response over 
    #   the dataset that is explained by the linear fit.
    r2 = SSR / SST
    # Always increases when an additional attribute is included.
    # To be useful, a new attribute must significantly increase R2.

    # Mean Squared Error
    MSE = SSE / (numRows - numPredictors - 1)

    # Standard Error of Estimation
    #   Interpreted as the typical size of residuals 
    s = np.sqrt(MSE)
    # Can be lower or higher when another attribute is added to the model.

    return r2, s