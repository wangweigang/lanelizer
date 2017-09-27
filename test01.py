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

numOfHalfRow = np.int(numOfRow/2)
numOfHalfCol = np.int(numOfCol/2)

imagex = np.hsplit(x, [numOfHalfCol,numOfCol])
image0010 = np.vsplit(imagex[0], [numOfHalfRow,numOfRow])
image0111 = np.vsplit(imagex[1], [numOfHalfRow,numOfRow])


x0 = np.concatenate((image0010[0],image0010[1]), axis=0)
x1 = np.concatenate((image0111[0],image0111[1]), axis=0)

print("x0.shape1:", x0.shape)
print("x1.shape1:", x1.shape)
x1 = np.array([]).reshape(4,0)
print("x1.shape2:", x1.shape)

z  = np.concatenate((x0,x1), axis=1)

u = np.array([image0010[0],image0010[1]])


print (x)
print (image0010[0])
print (image0010[1])
print (image0111[0])
print (image0111[1])
print ("z=\n",z)
print ("u=\n",u)

print (u[0])
