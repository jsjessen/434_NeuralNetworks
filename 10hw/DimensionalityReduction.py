#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 10
# Due 2019-11-12

import pandas as pd
import numpy as np

def dimensionalityReduction(dataPath, inputColumns):

    # Read data from file
    # Read CSV:  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    # DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    inputData = pd.read_csv(dataPath, usecols=inputColumns)

    # Each column represents a variable, while the rows contain observations.
    covarianceMatrix = np.cov(inputData, rowvar=False)

    # If we arrange the eigenvectors in W so that their 
    # eigenvalues λ1...λd are in decreasing order of magnitude, 
    # then the components of z, z = wx, 
    # are called “principle components” (PCs). 
    eigenvalues = np.linalg.eigvals(covarianceMatrix)

    # Sort eigenvalues largest to smallest (descending order)
    eigenvalues[::-1].sort()

    varianceExplained = []
    proportionVariances = []
    runningTotal = 0
    total = np.sum(eigenvalues)
    for eigVal in eigenvalues:
        runningTotal += eigVal
        varianceExplained.append((eigVal / total) * 100)
        proportionVariances.append((runningTotal / total) * 100)
    #   pov(k) = (Eig_1 + Eig_2 + ... + Eig_k) / (Eig_1 + Eig_2 + ... + Eig_k + ... + Eig_d)

    outputData = pd.DataFrame()
    outputData['Principal Component'] = list(range(1, len(eigenvalues)+1))
    outputData['Eigenvalues'] = eigenvalues
    outputData['Variance Explained'] = varianceExplained
    outputData['Proportion of Variance'] = proportionVariances

    return outputData