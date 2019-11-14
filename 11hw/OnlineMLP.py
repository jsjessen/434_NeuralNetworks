#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 11
# Due 2019-11-19

# Python Tutorial:      https://docs.python.org/3.7/tutorial/index.html
# Python Documentation: https://docs.python.org/3.7/index.html

# python 11hw/hw11.py

import math
import random
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Use a multilayer perceptron (MLP) with one hidden layer.
# Implement online training by non-linear regression.
def onlineMLP(dataset,
              inputWeights, 
              hiddenWeights, 
              learningRate,
              numIterations):

    # Input layer contains 3 nodes: bias=x0=1, x1, x2
    #
    #   XOR Input  | Output
    #  =========== | ======
    #   x0   x1	x2 |   r
    #  --- | ----- |  ---
    #   1  |  0	0  |   0
    #   1  |  0	1  |   1
    #   1  |  1	0  |   1
    #   1  |  1	1  |   0

    # Convert lists to numpy arrays for linear algebra.
    dataset = np.array(dataset)
    W = np.array(inputWeights)
    V = np.array(hiddenWeights)
    
    # Assuming all columns are inputs except the last, which is expected output.
    numRows, numColumns = dataset.shape
    lastIndex = numColumns - 1
    inputData = np.ones((numRows, numColumns)) # First column ones for bias node
    inputData[:,1:] = dataset[:, 0:lastIndex]
    outputData = dataset[:, lastIndex]

    # Hidden layer contains 3 nodes: bias=z0=1, z1, z2
    Z = np.ones(len(hiddenWeights))

    wShape = W.shape
    wRows, wColumns = wShape

    # Perform 1000 iterations with the example chosen randomly from the dataset.
    Error = np.zeros(numIterations)
    squares = 0
    for iteration in range(0, numIterations):
        randomIndex = random.randrange(numRows)
        X = inputData[randomIndex]
        r = outputData[randomIndex]

        # Transform the hidden nodes by sigmoid(wTx)
        for h in range(0, wRows):
            # Skip the bias node z[0]
            Z[h+1] = sigmoid(X.dot(W[h]))
        
        y = V.dot(Z)

        # Before each weight update, calculate the sum of squared residuals. 
        squares += math.pow(r - y, 2)
        Error[iteration] = squares / (iteration + 1)

        # Backpropagation
        c = learningRate * (r - y)
        changeV = c * Z
        changeW = np.zeros(wShape)
        for h in range(0 + 1, wRows + 1):
            changeW[h-1] = X.dot(c * V[h] * Z[h] * (1 - Z[h]))
        V = np.add(V, changeV)
        W = np.add(W, changeW)

    print('Final weights connecting input to hidden layer:')
    print(W)
    print('\n')

    print('Final weights connecting hidden layer to output:')
    print(V)
    print('\n')


    my_df = []
    for X, r in zip(inputData, outputData):
        # Transform the hidden nodes by sigmoid(wTx)
        for h in range(0, wRows):
            # Skip the bias node z[0]
            Z[h+1] = sigmoid(X.dot(W[h]))
        
        y = V.dot(Z)
        print("{} XOR {} = {} | y = {}".format(int(X[1]), int(X[2]), r, round(y,2)))
        d = { 'z1': Z[1], 'z2': Z[2] }
        my_df.append(d)

    Z = pd.DataFrame(my_df)
    print('\n')
    print(Z)

    sum1 = np.sum(Z.iloc[0])
    sum2 = np.sum(Z.iloc[1])
    decisionBoundary = np.mean([sum1, sum2])
    margin = sum1 - decisionBoundary

    print('Decision Boundary = {}'.format(round(decisionBoundary, 3)))
    print('Margin = {}'.format(round(margin, 3)))



    # Create semilog plot of convergence.

    # Report the final weight vectors and predicted y value for each example.

    # With the final weights, calculate and report the values z1 and z2
    # for each example in the dataset.

    # Use the values z1 and z2 associated with examples (0,0) and
    # (0,1) to calculate the bias of a decision boundary with equal
    # margins for the 2 classes. Report the margins and include a
    # plot of feature space with the decision boundary and location of
    # features associated with examples in the dataset.

    return Error, Z, decisionBoundary
