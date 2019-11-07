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

    # numInputs = len(inputColumns)
    # numRows = len(data.index)

    sig = np.cov(data)
    w, v = np.linalg.eig(sig)

    print(sig)
    print('-'*42)
    print(w)
    print('-'*42)
    print(v)

    # %columns of V are the eigenvectors
    # %D is diagonal
    # %eigenvalues on diagonal are in increasing order
    # %invert order and store in array eigenvals
    # d = length(eigenvals)
    # for k=1:d
    # pov(k) = <pic>
    # plot(pov)
    # %index of pov array will be used as x coordinate


    # Calculate principal components of attributes in glass data short.csv
    # (Do not include the 10th column, which is class labels)

    # Report the eigenvalues ranked by decreasing magnitude. 

    # Calculate PoV for all eigenvalues and plot.

    return