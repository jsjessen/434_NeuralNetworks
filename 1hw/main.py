#!/usr/bin/env python3

# James Jessen
# 2019-09-02
# CptS 434 - Assignment 1

import pandas

datafile = 'data/Cereals.csv'

target = 'Rating'

predictors = ['Sugars','Fiber']
# predictors.append('Protein')
# predictors.append('Fat')
# predictors.append('Sodium')

attributes = predictors.copy()
attributes.append(target)

#==============================================================================

# Read data from file
data = pandas.read_csv(datafile, usecols=attributes)

# Write a code for regression of nutritional rating vs sugar and fiber. 
# Train with example from Cereals dataset on class web page
# Report bias and slopes for predictors sugar and fiber
# Calculate sum of square residuals, R2, and standard error of estimation, s.




# Consider protein, fat, and sodium separately as a third attribute, 
# in addition to sugar and fiber, to predict the nutritional rating cereals.
# Report the change in R2 and s relative to sugar and fiber only.