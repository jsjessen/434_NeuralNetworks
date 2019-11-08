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
    data = pd.read_csv(dataPath, usecols=inputColumns)

    # Each column represents a variable, while the rows contain observations.
    covarianceMatrix = np.cov(data, rowvar=False)
    eigenvalues = np.linalg.eigvals(covarianceMatrix)

    # Sort eigenvalues largest to smallest (descending order)
    eigenvalues[::-1].sort()

    proportionVariance = []
    runningTotal = 0
    total = np.sum(eigenvalues)
    for eigVal in eigenvalues:
        runningTotal += eigVal
        proportionVariance.append(runningTotal / total)
    #   pov(k) = (Eig_1 + Eig_2 + ... + Eig_k) / (Eig_1 + Eig_2 + ... + Eig_k + ... + Eig_d)

    return eigenvalues, proportionVariance