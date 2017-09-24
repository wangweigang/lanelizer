# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 21:07:02 2017

@author: wang
"""

import numpy as np

x = np.arange(20.0).reshape(4, 5)

# 0    1   2   3   4
# 5    6   7   8   9
# 10  11  12  13  14
# 15  16  17  18  19

numOfRow,numOfCol = x.shape

numOfHalfCol = np.int(numOfCol/2)

y = np.hsplit(x, [numOfHalfCol,numOfCol])

z = np.concatenate((y[0],y[1]), axis=1)
print (x)
print (y[0])
print (y[1])
print (z)
