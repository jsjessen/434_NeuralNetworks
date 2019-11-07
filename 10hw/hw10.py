#!/usr/bin/env python3

# James Jessen
# CptS 434 - Assignment 10
# Due 2019-11-12

# Python Tutorial:      https://docs.python.org/3.7/tutorial/index.html
# Python Documentation: https://docs.python.org/3.7/index.html

# python 10hw/hw10.py > 10hw/output.txt

import pandas as pd
import numpy as np
from DimensionalityReduction import dimensionalityReduction as dr

dataPath = 'data/glassDataWithHeader.csv'
inputColumns = list(range(0,8))
precision = 2

dr(dataPath, inputColumns)