#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 4
# Due 2019-10-01

import pandas as pd
import numpy as np

def perceptronClassification(dataPath, inputColumns, classColumn):
    # print("Input Columns: {}".format(inputColumns))
    # print("Class Column: {}".format(classColumn))

    columns = inputColumns.copy()
    columns.append(classColumn)

    # Read data from file
    # Read CSV:  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    # DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    data = pd.read_csv(dataPath, usecols=columns)

    data.loc['actual']
    data.loc['predicted']

    numInputs = len(inputColumns)
    numRows = len(data.index)

    V = np.ones((numRows, 1 + numInputs)) # First column ones for bias node
    V[:,1:] = data.iloc[:,inputColumns].values
    
    classifications = data.iloc[:,classColumn].values

    # Solve the normal equation
    w = np.linalg.solve(V.T.dot(V), V.T.dot(classifications))

    fits = V.dot(w)
    yAvg = np.mean(classifications)

    # Sum of Squares Regression
    #   Measures variability of fit from mean response.
    ssrDiff = fits - yAvg
    SSR = np.sum(ssrDiff.dot(ssrDiff))

    # Sum of Squares Error
    #   Measures variability of response from all other sources after the linear relationship 
    #   between response and attributes has been accounted for.
    residuals = classifications - fits # Deviations predicted from actual empirical values of data
    SSE = np.sum(residuals.dot(residuals))

    # Sum of Squares Total 
    #   The sum of the squared differences of each observation from the overall mean.
    #   Identity: SST = SSR + SSE
    delY = classifications - yAvg
    SST = np.sum(delY.dot(delY))

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

    return classifications, fits, rSq