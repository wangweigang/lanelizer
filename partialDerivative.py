import numpy as np
import cv2


def partialY(matrix, dy=1):

  # Row -> y
  # remove first few rows
  matrixTemp = np.delete(matrix, np.s_[0:dy:1], 0)
  aRow = matrix[-dy:,:]
  matrixTemp = np.append(matrixTemp, aRow, 0)
  return matrix - matrixTemp
  
def partialX(matrix, dx=1):   
  # remove first few columns
  matrixTemp = np.delete(matrix, np.s_[0:dx:1], 1)
  aCol = matrix[:,-dx:]
  print ('partialX: ', matrixTemp.shape, aCol.shape)
  matrixTemp = np.append(matrixTemp, aCol, 1)
  return matrix - matrixTemp
