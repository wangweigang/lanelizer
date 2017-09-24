import numpy as np
import cv2
import shake
import sys
#
#  process image data: find edge, reduce data by mask
#
def imageProc(image, roiVertice, verticeI):
        
    
    image = windowFront(image, roiVertice)  
    # cv2.imshow('Test', image) # output shown on screen is RGB image, therefore input is BGR
    # 
    # use Clahe to raise contrast
    #
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h,l,s = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l     = clahe.apply(l)
    
    # kernelc = np.ones((11,11), np.uint8)
    # kernelo = np.ones((6,6), np.uint8)
    #s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, kernelc)   # close separate white spots
    #s = cv2.morphologyEx(s, cv2.MORPH_OPEN,  kernelo)   # delete separate white spots
  
    image = cv2.merge((h,l,s))
    image = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
    # cv2.imshow('After CLAHE Correction', image)
     
    if False:    # masking colors
        
      if True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # catch good white lines under the sun
        lower = np.array([20-4,2,250-4])  
        upper = np.array([30+4,6+4,254])   
        # lower = np.array([30-1,6-1,253-1])  
        # upper = np.array([30+1,6+1,253+1])   
        mask  = cv2.inRange(image, lower, upper)
        mask[mask > 0] = 255
   
        # catch good yellow lines under the sun
        lower    = np.array([23-4,64-4,246-4])  
        upper    = np.array([25+4,106+4,253])   
        maskTemp = cv2.inRange(image, lower, upper)
        maskTemp[maskTemp > 0] = 255
        mask     = np.add(mask, maskTemp, dtype=np.uint8)
        
        # catch good white lines under steet light
        lower    = np.array([108-4,21-8,144-4])  
        upper    = np.array([113+4,57+4,206+22])   
        maskTemp = cv2.inRange(image, lower, upper)
        maskTemp[maskTemp > 0] = 255
        mask     = np.add(mask, maskTemp, dtype=np.uint8)
        
        # catch dirty yellow lines under steet light
        lower    = np.array([14-4, 121-4, 83-4])  
        upper    = np.array([22+4, 185+8, 157+4])   
        maskTemp = cv2.inRange(image, lower, upper)
        maskTemp[maskTemp > 0] = 255
        mask     = np.add(mask, maskTemp, dtype=np.uint8)
      
      # # yellow in tunnel
      # lower    = np.array([20-1,  97-10, 51-2])   
      # upper    = np.array([21+4, 100+2,  66+5])  
      # maskTemp = cv2.inRange(image, lower, upper)
      # maskTemp[maskTemp > 0] = 255
      # mask = np.add(mask, maskTemp, dtype=np.uint8)
      
      # # dirty white in tunnel
      # lower    = np.array([ 16, 133,  71])   
      # upper    = np.array([ 19, 175, 130])  
      # maskTemp = cv2.inRange(image, lower, upper)
      # maskTemp[maskTemp > 0] = 255
      # mask = np.add(mask, maskTemp, dtype=np.uint8)
      
      # # dirty white1 in tunnel
      # lower    = np.array([106,  48,  47])   
      # upper    = np.array([109,  60,  59])  
      # maskTemp = cv2.inRange(image, lower, upper)
      # maskTemp[maskTemp > 0] = 255
      # mask = np.add(mask, maskTemp, dtype=np.uint8)
      
      # # catch dirty dashed white lines
      # lower    = np.array([24-2,   9-0, 200+4])     
      # upper    = np.array([60-24, 34+8, 236+32])   
      # maskTemp = cv2.inRange(image, lower, upper)
      # maskTemp[maskTemp > 0] = 255
      # mask = np.add(mask, maskTemp, dtype=np.uint8)
      
      # # catch dirty1 dashed white lines
      # lower    = np.array([20,  8, 180])     
      # upper    = np.array([30, 21, 237])   
      # maskTemp = cv2.inRange(image, lower, upper)
      # maskTemp[maskTemp > 0] = 255
      # mask = np.add(mask, maskTemp, dtype=np.uint8)

 
      # # catch dashed white lines
      # lower    = np.array([24-5, 27-5,  244-5])
      # upper    = np.array([24+5, 27+5, 244+5])
      # maskTemp = cv2.inRange(image, lower, upper)
      # maskTemp[maskTemp > 0] = 255
      # mask = np.add(mask, maskTemp, dtype=np.uint8)

      # # catch dashed very white lines
      # lower    = np.array([19-12, 2,   249-9])
      # upper    = np.array([30+15, 8+4, 251+4])
      # maskTemp = cv2.inRange(image, lower, upper)
      # maskTemp[maskTemp > 0] = 255
      # mask = np.add(mask, maskTemp, dtype=np.uint8)

      # # catch dirty yellow lines under the sun
      # lower = np.array([22-4,  51, 171-6])  
      # upper = np.array([25+4, 109, 227])   
      # maskTemp  = cv2.inRange(image, lower, upper)
      # maskTemp[maskTemp > 0] = 255   
      # mask = np.add(mask, maskTemp, dtype=np.uint8)
      
      # # catch white lines under shadow
      # lower = np.array([107,  48,  69])  
      # upper = np.array([112,  61, 118])   
      # maskTemp  = cv2.inRange(image, lower, upper)
      # maskTemp[maskTemp > 0] = 255   
      # mask = np.add(mask, maskTemp, dtype=np.uint8)
      
      # # catch dirty white lines
      # lower = np.array([ 20,   1, 147])  
      # upper = np.array([111,  20, 224])   
      # maskTemp  = cv2.inRange(image, lower, upper)
      # maskTemp[maskTemp > 0] = 255   
      # mask = np.add(mask, maskTemp, dtype=np.uint8)
      
      # cv2.imshow('before shake', np.array(mask))

      # shake a few times
      # for i in range(1, 2): mask = shake.mask(mask, 1)
      #mask = maskShake(mask, 3)
   
        
  ##    kernel1 = np.ones((11,11), np.uint8)
  ##    kernel2 = np.ones((6,6), np.uint8)
  ##    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  kernel1)  # close separate white spots
  ##    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel2)   # delete separate white spots
      # cv2.imshow('After shake', np.array(mask))
   
      # mask = cv2.GaussianBlur(mask, (3,3), 0)
      # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)  # close white spots
      # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel1)  # delete separate white spots
      # mask = cv2.GaussianBlur(mask, (3,3), 0)
      # image[mask > 0] = [0,0,0]
      #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
      cv2.imshow('Gray befor Canny', cv2.resize(image,(417,155)))
      
      if True:
        # get V channel mean of low half of the window
        nLowhalf       = np.uint32(len(image[:,1,1])/2)
        intensityLight = np.mean(image[nLowhalf:,:,2])
        # print('                     intensity: ',intensityLight)

        # set the masked region to black for all channels of image: yellow middle line is marked as black

        if intensityLight < 52:
            # in dark
            image[mask > 0] = [255,255,255] 
        else:
            # bright
            image[mask > 0] = [0,0,0]
      
      image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
      cv2.imshow('Color befor Canny', cv2.resize(image,(417,155)))
    elif True:     
      if True:
        # make S channel of HLS (RGB2HLS for a BGR image)as image for Canny: good for yellow lines under the sun (both RGB2HLS and BGR2HLS work, BGR2HLS is better)
        # make S channel of HSV (RGB2HSV for a BGR image) as image for Canny: good for yellow lines under the sun (both RGB2HLS and BGR2HLS work)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        image = extractColor(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        _,l,image1 = cv2.split(image)
        #print ("   hls=", np.min(l),np.max(l), np.min(image1),np.max(image1))
        
        # use my filter
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#        _,image,_ = cv2.split(image)     
        
        #        
        # make threshold intensity dependent
        #
#        mask = np.zeros_like(l)
#        cv2.fillPoly(mask, verticeI, 255)
#        intensityLight = np.mean(l)
#        #print("                     intensityLight: ", intensityLight)
#        tempVar = np.min([250.0, intensityLight*1.8])
#        thresholdSH = np.uint8(tempVar)
#        thresholdSL = np.uint8(intensityLight*1.1)
#        image[l>thresholdSH] = 255
#        image[l<thresholdSL] = 0
#        
#        kernelc = np.ones((4,4), np.uint8)
#        #kernelo = np.ones((4,4), np.uint8)
#        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelc)  # close/expand separate white spots
        #image = cv2.morphologyEx(image, cv2.MORPH_OPEN,  kernelo) # delete/blacken separate white spots
      else:
        # make GRAY image
        image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
      
      cv2.imshow('my extraction',    cv2.resize(image1,(417,155)))
      # cv2.imshow('Gray befor Canny', cv2.resize(image1,(417,155)))
      #stop()
    else:
# use cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) → dst        
      laplacian = cv2.Laplacian(image, cv2.CV_64F)
      sobelx    = cv2.Sobel(image, cv2.CV_64F, 1,0, ksize=5)  # x
      sobely    = cv2.Sobel(image, cv2.CV_64F, 0,1, ksize=5)  # y
      sobel     = cv2.Sobel(image, cv2.CV_64F, 1,1, ksize=5)  # y
      cv2.imshow('laplacian', cv2.resize(laplacian,(417,155)))
      cv2.imshow('sobelx',    cv2.resize(sobelx,(417,155)))
      cv2.imshow('sobely',    cv2.resize(sobely,(417,155)))
      cv2.imshow('sobel',     cv2.resize(sobel,(417,155)))
      stop()
      
      
    return image   # RGB image



def extractColor(image):
    
    # image[...,0] = (1-np.sin(0.5*3.14159*image[...,1]/255))*255  
    # not get yellow strong sun und shadow, tunnel
    image[...,1] = np.sin(0.5*3.14159*image[...,1]/255)*255  
    image[...,2] = np.sin(0.5*3.14159*image[...,2]/255)*255
#    
    image = np.uint8(image)
    
    if False:
      s = np.zeros_like(image[...,0], dtype=np.float16)
      l = np.zeros_like(image[...,0], dtype=np.float16)
      numOfRow, numOfCol = s.shape
      for m in range(numOfRow):
          for n in range(numOfCol):
              b,g,r   = image[m,n,:]
              b1      = np.float16(b)/255
              # yellow -> black; white -> dark grey
              # g1      =1- np.cos(0.5*3.14159*np.float16(g)/255)   
              # r1      =1- np.cos(0.5*3.14159*np.float16(r)/255)
              # yellow -> black; white -> dark grey
              # g1      = np.cos(0.5*3.14159*np.float16(g)/255)  
              # r1      =1- np.cos(0.5*3.14159*np.float16(r)/255)
              # yellow -> dark grey; white -> dark grey
              # g1      = np.sin(0.5*3.14159*np.float16(g)/255)   
              # r1      = np.cos(0.5*3.14159*np.float16(r)/255)
              # yellow -> light grey; white -> dark grey
              # g1      = np.cos(0.5*3.14159*np.float16(g)/255)   
              # r1      = np.sin(0.5*3.14159*np.float16(r)/255)
              # no good
              # g1      = 1-np.cos(0.5*3.14159*np.float16(g)/255)   
              # r1      = np.sin(0.5*3.14159*np.float16(r)/255)
              # yellow -> light grey; white -> dark grey
              # g1      = np.cos(0.5*3.14159*np.float16(g)/255)   
              # r1      = 1-np.sin(0.5*3.14159*np.float16(r)/255)
              # yellow -> light grey; white -> dark grey: Canny and Hough work
              # g1      = np.sin(0.5*3.14159*np.float16(g)/255)   
              # r1      = 1-np.sin(0.5*3.14159*np.float16(r)/255)
              # yellow -> light grey; white -> dark grey
              # g1      = np.cos(0.5*3.14159*np.float16(g)/255)   
              # r1      = np.cos(0.5*3.14159*np.float16(r)/255)
              # looks good bud Canny does not get it
              g1      = np.sin(0.5*3.14159*np.float16(g)/255)   
              r1      = np.sin(0.5*3.14159*np.float16(r)/255)
              
              
              vmax   = np.float16(np.max([r1,g1,b1])) 
              vmin   = np.float16(np.min([r1,g1,b1])) 
              li     = np.float16(vmax+vmin)/2 
              l[m,n] = li
              dv     = (vmax-vmin) 
              li0    = np.abs(2*li-1)
              if (dv == 0) or (li0==1.0):
                  s[m,n] = 0.0
              else:
                s[m,n] = dv/(1.0-li0)
              
              # vs     = np.float16(vmax+vmin)
              # if li == 0:
                  # s[m,n] = 0.0
              # elif li>0.5:
                  # s[m,n] = dv/(2.0-vs)
              # else:
                  # s[m,n] = dv/vs
              continue
      s = 255*(s-np.min(s))/(np.max(s)-np.min(s))
      l = 255*(l-np.min(l))/(np.max(l)-np.min(l))
      # print ("my hls=", np.min(l),np.max(l), np.min(s),np.max(s))
       
    return image

# mask image data
#
def windowFront(image, roiVertice):
    # from sendet
    #make mask all zeros with size of image
    # image = np.array(image)
    # print('wdecw: ',image.shape[0],image.shape[1])
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    #filling pixels inside the polygons defined by "roiVertice" with the fill color (can be convex or concave)
    #cv2.fillPoly(mask, roiVertice, [255,255,255])
    cv2.fillPoly(mask, roiVertice, 255)

    #returning the image only where mask pixels are nonzero
    # masked = cv2.bitwise_and(image, mask)
    # cv2.imshow('Mask', mask)
    # print('size2: ', mask.shape, image.shape)
    image[mask == 0] = 0
    # zzz = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    # cv2.imshow('Image masked', zzz)
    # stop()
    return image

#
# stop by click close window  
#        
def stop():
    cv2.waitKey(0)
    # input('Any key to stop ... ...')
    cv2.destroyAllWindows()
    sys.exit()
    
def pause():
    print("Press any key to continue ... ...")
    cv2.waitKey(0)