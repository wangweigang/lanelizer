import cv2
import numpy as np
import sys

def edgeProc(image):

    if True: # for image of one channel
        # cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) → dst
        # src – input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
        # dst – output image of the same size and type as src.
        # ksize – Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero’s and then they are computed from sigma* .
        # sigmaX – Gaussian kernel standard deviation in X direction.
        # sigmaY – Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height , respectively (see getGaussianKernel() for details); to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
        # borderType – pixel extrapolation method (see borderInterpolate for details).
        # for color image
        image = cv2.GaussianBlur(image, (5,5), 0)

        # edge detection
        # cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]])     # Parameters
        # image	8-bit input image.
        # edges	output edge map; single channels 8-bit image, which has the same size as image .
        # threshold1	first threshold for the hysteresis procedure.
        # threshold2	second threshold for the hysteresis procedure.
        # apertureSize	aperture size for the Sobel operator.
        # L2gradient	a flag, indicating whether a more accurate L2 norm
        # image = cv2.Canny(image, threshold1 = 200, threshold2=300)

        # no dash
        # image = cv2.Canny(image, threshold1=20, threshold2=222)
        # get white dash,
        # image = cv2.Canny(image, threshold1=20, threshold2=166, apertureSize=3)
        # no whitening, get dash, no yellow, too many line
        # image = cv2.Canny(image, threshold1=20, threshold2=144, apertureSize=3, L2gradient=True)
        # whitening, get dash
        # image = cv2.Canny(image, threshold1=111, threshold2=333, apertureSize=3, L2gradient=True)
        # whitening, get dash, get bright yellow
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # sValueMax = np.max(image)
        #print('sValueMax=', sValueMax)
        #image = image * 255.0/sValueMax
# nor working        
#        medianImg     = np.median(image)
#        sigma         = 0.11
#        thresholdLow  = int(max(0,   (1.0 - sigma) * medianImg))
#        thresholdHigh = int(min(255, (1.0 + sigma) * medianImg))
#        print("threshold=", thresholdLow,thresholdHigh)
        # image = cv2.Canny(image, threshold1=thresholdLow, threshold2=thresholdHigh, apertureSize=3, L2gradient=True)
        image = cv2.Canny(image, threshold1=66, threshold2=255, apertureSize=3, L2gradient=True)

#       do sobel on y (partial derivative along y) to emphasize the      
#        image = cv2.Sobel(image, cv2.CV_64F, 0,1, ksize=5)
#        image = image.astype(np.uint8)
#        cv2.imshow('sobel',     cv2.resize(image, (417,155)))
#        laplacian = cv2.Laplacian(image, cv2.CV_64F)
#        sobelx    = cv2.Sobel(image, cv2.CV_64F, 1,0, ksize=5)  # x
#        sobely    = cv2.Sobel(image, cv2.CV_64F, 0,1, ksize=5)  # y
#        sobel     = cv2.Sobel(image, cv2.CV_64F, 1,1, ksize=5)  # y
#        cv2.imshow('laplacian', cv2.resize(laplacian,(417,155)))
#        cv2.imshow('sobelx',    cv2.resize(sobelx,(417,155)))
#        cv2.imshow('sobely',    cv2.resize(sobely,(417,155)))
#        cv2.imshow('sobel',     cv2.resize(sobel, (417,155)))
#        stop()

    else: # for color image
 
        # get no dash
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.Canny(image, threshold1 = 450, threshold2=500, apertureSize=3)

    return image
        
#
# stop by click close window  
#        intenMax = image[:,i,1]
def stop():
    cv2.waitKey(0)
    # input('Any key to stop ... ...')
    cv2.destroyAllWindows()
    sys.exit()
        
def pause():
    print("Press any key to continue ... ...")
    cv2.waitKey(0)
