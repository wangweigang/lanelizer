import numpy as np
import cv2


def mask(mask, shift):

    if True:
        # remove first few rows
        maskTemp = np.delete(mask, np.s_[0:shift:1], 0)

        aRow = np.zeros((shift, mask.shape[1]), dtype=np.uint8)
        maskTemp = np.append(maskTemp, aRow, 0)
        mask = np.add(maskTemp, mask, dtype=np.uint8)
        

        # remove last few rows
        lastRow = mask.shape[0]
        maskTemp = np.delete(mask, np.s_[lastRow-shift:lastRow:1], 0)

        aRow = np.zeros((shift, mask.shape[1]), dtype=np.uint8)
        maskTemp = np.append(aRow, maskTemp, 0)
        mask = np.add(maskTemp, mask, dtype=np.uint8)



    if False:   
        # remove first few columns
        maskTemp = np.delete(mask, np.s_[0:shift:1], 1)

        aCol = np.zeros((mask.shape[0],shift), dtype=np.uint8)
        maskTemp = np.append(maskTemp, aCol, 1)
        mask = np.add(maskTemp, mask, dtype=np.uint8)



        
        
        # remove last few columns
        lastCol = mask.shape[1]
        maskTemp = np.delete(mask, np.s_[lastCol-shift:lastCol:1], 1)

        aCol = np.zeros((mask.shape[0],shift), dtype=np.uint8)
        maskTemp = np.append(aCol, maskTemp,  1)
        mask = np.add(maskTemp, mask, dtype=np.uint8)

        # cv2.imshow('After shake4', np.array(mask))
        # keyPressed = cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return mask

