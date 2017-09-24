import numpy as np
#from numpy import ones, vstack
#from numpy.linalg import lstsq
#from statistics import mean
import cv2
import os, sys
import matplotlib.pyplot as plt
import partialDerivative as pd
import warnings

#
#  find and draw lines / lanes 
#    
def lineFind (imageOriginal, image):

    lines = []
    # no blurring, too few lines
    # image = cv2.GaussianBlur(image, (3,3), 0)
    
    #
    # http://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html?highlight=cv2.houghlinesp#cv2.HoughLinesP    #                               rho   theta   thresh  min length, max gap:        

    # cv2.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) → lines
    # image – 8-bit, single-channel binary source image. The image may be modified by the function.
    # lines – Output vector of lines. Each line is represented by a 4-element vector (x_1, y_1, x_2, y_2) , where (x_1,y_1) and (x_2, y_2) are the ending points of each detected line segment.
    # rho – Distance resolution of the accumulator in pixels.
    # theta – Angle resolution of the accumulator in radians.
    # threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes ( >\texttt{threshold} ).
    # minLineLength – Minimum line length. Line segments shorter than that are rejected.
    # maxLineGap – Maximum allowed gap between points on the same line to link them.

    # no whitening: get a few long line and curbs, no yellow line
    # lines = cv2.HoughLinesP(image, 1, np.pi/180, 180, np.array([]), 222, 11)
    # get a few whitening lines
    # lines = cv2.HoughLinesP(image, 1, np.pi/180, 180, np.array([]), 55, 0)
    # lines = cv2.HoughLinesP(image, 0.8, np.pi/100, 222, np.array([]), 111, 11)
    # lines = cv2.HoughLines2(image, 0.8, np.pi/100, 222, np.array([]), 111, 11)

    if False:       # my way  
        # use HSV and horizontal line by line (may, be Kalman filter)
        lineFinal = lineByIntensity(imageOriginal)
    else:           # Hough line

        if True:
            # make white then black strips to mask edge image
            # 111111111111111, 0101010101, 110110110110110, ...
            numOfRow, numOfCol = image.shape
            mask               = np.zeros_like(image)
            for i in range(0,numOfRow,2):
                mask[i:i+1,:] = 255
                #mask[i:i+2,:] = 255
            for j in range(0,numOfCol,2):
                mask[:,j:j+1] = 255
                #mask[:,j:j+2] = 255
            image[mask>1] = 0


        # image = cv2.GaussianBlur(image, (7,7), 0)
        if False:
            # do normal HoughLines: not working
            linez = cv2.HoughLines(image,1,np.pi/180,30)
#           array([[[ 370.        ,    0.9075712 ]],           
#                  [[ 378.        ,    0.89011788]],         
#                  [[ 367.        ,    0.92502451]]], dtype=float32)            

            #print("Num of lines=", len(linez), linez)
            lines = np.array(linez)
            linet = np.array([]).reshape(0,1,4)
            for rho,theta in lines[:,0,:]:
                if (rho > 0.0) and (theta > 0) and (theta < 1.57): 
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = 0
                    x2 = 803
                    y1 = y0 + a/b*(x1-x0)
                    y2 = y0 + a/b*(x2-x0)
                    
                    # for bad y1
                    if  y1 > 301:
                        y1 = 301
                    elif y1 < 0:
                         y1 = 0
                    x1 = x0 + b/a*(y1-y0)
                    # for bad y2
                    if  y2 > 301:
                        y2 = 301
                    elif y2 < 0:
                        y2 = 0
                    x2 = x0 + b/a*(y2-y0)
                    # for bad x1
                    if  x1 > 803:
                        x1 = 803
                    elif x1 < 0:
                        x1 = 0
                    y1 = y0 + a/b*(x1-x0)
                    # for bad x2
                    if x2 > 803:
                        x2 = 803
                    elif x2 < 0:
                        x2 = 0
                    y2 = y0 + a/b*(x2-x0)

                        
                    # print("theta r x:",theta, rho, [x1,x2,y1,y2])                    
                    linet = np.vstack([linet, [[[x1,x2,y1,y2]]]])

            lineFinal = linet.astype(int)
 
#            lineFinal = np.copy(linet)
            # lines = map(np.int, np.copy(linet))
            # lines = np.array(lines)
            # print("lines=",  lineFinal.shape)
        elif False:
# for sieve 11111111111111 (road.15.png) 
#            lineFinal = cv2.HoughLinesP(image=image, rho=0.2, 
#                        theta=np.pi/100, threshold=20, lines=11, 
#                        minLineLength=99, maxLineGap=22)
#
# for sieve 01010101010101 (road.15.png) 
            lineFinal = cv2.HoughLinesP(image=image, rho=1, theta=np.pi/110,
                             threshold=22, lines=np.array([]),
                             minLineLength=11, maxLineGap=33)
#
#           lines format
#           array([[[259,  79, 259,  31]],
#                  [[205, 309, 345, 201]],
#                  [[819, 159, 825, 211]]], dtype=int32)#        print('Line found:                  ', len(lines),lines.shape, lines)
        else:
# segment the image to 2,4,6,8, ...
            numOfRow,numOfCol = image.shape
            numOfHalfCol = np.int(numOfCol/2)

            imageSub     = np.hsplit(image, [numOfHalfCol,numOfCol])
            if False:
                # for 11111111111111111 road04.png 
    #            lines0 = cv2.HoughLinesP(image=imageSub[0], rho=0.5, 
    #                        theta=np.pi/100, threshold=66,   
    #                        minLineLength=99, maxLineGap=11)
                # for 11111111111111111 road14.png  
                lines0 = cv2.HoughLinesP(image=imageSub[0], rho=0.5, 
                            theta=np.pi/111, threshold=11,   
                            minLineLength=66, maxLineGap=70)
                # for 11111111111111111 road14.png  
                lines1 = cv2.HoughLinesP(image=imageSub[1], rho=1, 
                            theta=np.pi/100, threshold=11,   
                            minLineLength=44, maxLineGap=44)
            else:
              # make 4 segments00 01;10 11
                # for 010101010101 road14.png  
                lines00 = cv2.HoughLinesP(image=imageSub[0], rho=1, 
                            theta=np.pi/111,  threshold=11,   
                            minLineLength=55, maxLineGap=66)
                # for 010101010101 road14.png  
                lines01 = cv2.HoughLinesP(image=imageSub[1], rho=1, 
                            theta=np.pi/100, threshold=11,   
                            minLineLength=44, maxLineGap=55)
              
            # shift x -> x + numOfHalfCol
            lines01[:,:,0] = lines01[:,:,0] + numOfHalfCol
            lines01[:,:,2] = lines01[:,:,2] + numOfHalfCol
            lineFinal = np.concatenate((lines00,lines01), axis=0)
#            print('Line found: ', len(lines0),lines0.shape, lines0.size)
#            print('Line found: ', len(lines1),lines1.shape, lines1.size)

#        stop()
        
    #print (minLineLength, len(lines))

    # too many lines
    # lines = cv2.HoughLinesP(image, 1, np.pi/180, 180, np.array([]), 10, 22)
    # lines = cv2.HoughLinesP(image, 1, np.pi/18, 10, 22, 9)
#    print('Line found:                  ', len(lineFinal),lineFinal.shape, lineFinal.size)
    
    return lineFinal



def laneFind(lines, roiVertice):
    #np.seterr(rank='ignore')
    warnings.simplefilter('ignore', np.RankWarning)    
    
    lineGood = []
    # print (roiVertice)
    vertices = roiVertice[0]
    
    #lineTest = np.asarray(lines)

    # return if no line found
#    for lines size (34,1,4)
#    print("line: ", len(lines) )    #=34
#    print("line: ", lines.shape )   #=(34,1,4)
#    print("line: ", lines.size)     #=136
    
    if lines.size < 4: return lineGood
    
    # filtered out by line positions
    
    for coords in lines:
        coords = coords[0]
        try:
            # coords[0] = # x1
            # coords[1] = # y1
            # coords[2] = # x2
            # coords[3] = # y2
            
            if False:    # take all the lines found
                lineGood.append(coords)
                # print ('                        coords=',coords)
            else:
                #    vertices (ROI)
                #    (0)-------(1)
                #     |         |
                #     |         |
                #    (3)-------(2)
                
                if coords[2]==coords[0]:
                    slope = 1.0e11
                else:
                    slope = (coords[3]-coords[1]) / (coords[2]-coords[0])

                if   (np.abs(slope) < 0.5) and (0.5*(coords[1]+coords[3]) < 0.5*(vertices[0][1]+vertices[1][1])+77):
                    # ignore horizontal lines at the top
                    continue
                elif (np.abs(slope) < 0.5) and (0.5*(coords[1]+coords[3]) > 0.5*(vertices[2][1]+vertices[2][1])-77):
                    # ignore horizontal lines at the bottom
                    continue
                elif (np.abs(coords[2]-coords[0]) < 33) and (0.5*(coords[0]+coords[2]) < vertices[0][0]+44):
                    # ignore vertical lines at the left
                    continue
                elif (np.abs(coords[2]-coords[0]) < 33) and (0.5*(coords[0]+coords[2]) > vertices[3][0]-44):
                    # ignore vertical lines at the right
                    continue
#                elif ((slope <  0.1) and (0.5*(coords[1]+coords[3]) < 0.3*(vertices[1][1]+vertices[2][1])) and (0.5*(coords[0]+coords[2]) < 0.4*(vertices[0][0]+vertices[1][0]))) or \
#                     ((slope > -0.3) and (0.5*(coords[1]+coords[3]) < 0.3*(vertices[0][1]+vertices[3][1])) and (0.5*(coords[0]+coords[2]) > 0.6*(vertices[0][0]+vertices[1][0]))):
                elif ((slope <  0.1) and (np.min((coords[1],coords[3])) < 0.2*0.5*(vertices[1][1]+vertices[2][1])) and (np.min((coords[0],coords[2])) < 0.1*0.5*(vertices[0][0]+vertices[1][0]))) or \
                     ((slope > -0.1) and (np.min((coords[1],coords[3])) < 0.2*0.5*(vertices[0][1]+vertices[3][1])) and (np.max((coords[0],coords[2])) > 1.9*0.5*(vertices[0][0]+vertices[1][0]))):
                    # ignore lines \ starting at upper-left  part
                    # ignore lines / starting at upper-right part
                    continue
           # elif 0.5*(coords[1]+coords[3]) < 0.5*(vertices[0][1]+vertices[3][1]):    

                    # continue
                #elif (np.absolute(slope) < 0.1) and (np.absolute(coords[0]-coords[1]) < 0.4*np.absolute(vertices[0][0]-vertices[1][0])):
                    # ignore short horizontal lines in the horizontal middle: it can be profile of a car in front of us
                    #continue
                elif abs(slope) < 0.001:
                    # ignore absolutely horizontal line
                    continue
                else:
                    lineGood.append(coords)
                 
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(str(e))
    
    lineGood = np.array(lineGood, dtype=np.uint16)

    return lineGood    # np.transpose(lineGood)



def lineGroup(lanes):
    # 
    # do line grouping/fitting
    #
    orderPoly  = 1     # 2  # 1
    point2Draw = 2     # 4  # 2
    try:
        laneTemp = np.copy(lanes)    
#        laneTemp = np.array(lanes, copy=True)    
        # laneTemp = lanes    
        lineFit    = np.array([], dtype=np.uint16).reshape(0,2*point2Draw)
        # lineFit = np.array([1,4], dtype=np.uint16)
        
        #print(len(lanes), lanes.shape, len(laneTemp), laneTemp.shape, )
         
         # numOfLines = len(lanes[:,0])
#        print("wewefwef",len(lanes))
#        print("xcvdfvdfv",len(laneTemp))
        
        # numOfLines = len(lanes)
        numOfRest  = len(laneTemp)
        # print("numOfRest=",numOfRest)
        i = 0
        j = i+1
        while i<numOfRest:
            l1          = laneTemp[i]
            lineCluster = np.array([l1])
            while j<numOfRest:
                # firstly check the slopes
                l2       = laneTemp[j]
                # print ("point line dist=", i, j, l1,l2, numOfRest)
                
                paralell = isParalell(l1,l2)
                # if the 2 lines are approximately paralell
                if paralell:
                    # then check the gap fro not too big
                    gap = gapBetweenLines(l1,l2)
                    # print("gap=", gap)
                    if gap < 300:
                        # then the distance
 
                        dist =        0.25*distPoint2Line([l1[0],l1[1]], l2)
                        dist = dist + 0.25*distPoint2Line([l1[2],l1[3]], l2)
                        dist = dist + 0.25*distPoint2Line([l2[0],l2[1]], l1)
                        dist = dist + 0.25*distPoint2Line([l2[2],l2[3]], l1)
                        
                        # print ("point line dist=", i, j, l1,l2, numOfRest, "%4d" % dist)
                        # pause()
                          # if isParalell(l1,l2) and (distBetweenParaLines(l1, l2)<11):
                            # check if l2 can be extended
                            
                        # if the 2 lines get small distance (close to each other)
                        if dist < 16.0:
                            lineCluster = np.append(lineCluster,[l2],axis=0)
        
                            # delete j row in l2
                            laneTemp  = np.delete(laneTemp, j, 0)
                            numOfRest   = numOfRest - 1
                            # print ("                       lineCluster growing=",numOfRest,len(lineCluster))
                        else:
                            j = j + 1
                    else:
                        j = j + 1
                else:
                    j = j + 1


            if len(lineCluster)==1:
                x    = np.concatenate((lineCluster[:,0],lineCluster[:,2]), 0)
                y    = np.concatenate((lineCluster[:,1],lineCluster[:,3]), 0)
                x    = np.array(x, dtype=np.float)
                y    = np.array(y, dtype=np.float)
                
                xmin = np.amin(x) 
                xmax = np.amax(x)
                if (xmax-xmin)>0:
                    xNew = np.linspace(xmin, xmax, point2Draw)
                    yNew = y[0] + (y[1]-y[0])*(xNew - x[0])/(x[1]-x[0])
                else:
                    xNew = np.linspace(xmin, xmax, point2Draw)
                    yNew = np.ones_like(xNew) * 0.5*(xmin+xmax)
                # print()
                lineFit = np.vstack([lineFit, [np.array([xNew,yNew], dtype=np.uint).flatten('C')]])
                    
            else:
                x    = np.concatenate((lineCluster[:,0],lineCluster[:,2]), 0)
                xmin = np.amin(x) 
                xmax = np.amax(x)
                # 
                y    = np.concatenate((lineCluster[:,1],lineCluster[:,3]), 0)
                mc   = np.polyfit(x, y, orderPoly)
#                print("fit=",len(x),x,y)
#                pause()
                xNew = np.linspace(xmin, xmax, point2Draw)
#                ymin = np.polyval(mc, xmin)
#                ymax = np.polyval(mc, xmax)
                yNew = np.polyval(mc, xNew)
#                ttt  = np.array([x,y])
#                rrr  = np.array([x,y]).flatten('F')
#                print("x y=", x,y )
#                print("ttt=", ttt )
#                print("rrr=", rrr)
#                rrr = np.uint([xNew,yNew]).flatten('F')  # F one col after another
#                www = np.uint([xNew,yNew]).flatten('C')
                # print("for lineFit: ", lineFit.shape, )
                lineFit = np.vstack([lineFit, [np.uint([xNew,yNew]).flatten('C')]])
                # print("lineFit: ", lineFit)
                # pause()
                
            i = i+1  
            j = i+1
    # print("                     laneTemp=",  numOfLines0, numOfRest, len(laneTemp[:,0]))
    # pause()
    except Exception as e:
        # print("line=", lines) 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(str(e))
        pass
    
    return lineFit      #np.array(lineFit, dtype=np.int)   # np.transpose(lineGood)



def distPoint2Line(point, line):
    
    dist = 1e4
    # casting for python list
    (point[0],point[1]) = map(np.float32, (point[0],point[1]))
    # castin for np array
    line = line.astype(np.float32)
 
    #print(point, line)
    d31  = line[3]-line[1]
    d20  = line[2]-line[0]
    
    if (np.abs(d31)>1.0e-4) or (np.abs(d20)>1.0e-4):   
        # [0]:x1; [1]:y1;  [2]:x2; [3];y2
        # print("dist=", d20, d31, d20*d20 + d31*d31,np.sqrt(d20*d20 + d31*d31))
        
        dist = np.abs(d31*point[0] - d20*point[1] + line[2]*line[1] - line[3]*line[0]) \
             / np.sqrt(d20*d20 + d31*d31)
        # stop()
        #print("point distance=", dist, line[3],line[1],line[2],line[0], (line[3]-line[1])**2, (line[2]-line[0])**2)
    return dist


def distBetweenParaLines(l1, l2):
    # only enter if l1 and l2 paralell
    
    # they are different lines
    if False:   # (not isParalell(l1,l2)):
        distance = 0
    else:
        a1 =  l1[3]-l1[1]
        b1 = -l1[2]+l1[0]
        c1 = -l1[0]*(l1[3]-l1[1]) - l1[1]*(l1[2]-l1[0])
        c2 = -l2[0]*(l2[3]-l2[1]) - l2[1]*(l2[2]-l2[0])
        a2 =  l2[3]-l2[1]
        b2 = -l2[2]+l2[0]
        if a1 == 0:
            distance = np.abs(c2/b2 - c1/b1)
        elif b1 == 0:
            distance = np.abs(c2/a2 - c1/a1)
        else:
            a  = a1*a2
            b  = b1*a2
            c1 = c1*a2
            c2 = c2*a1
            distance = np.abs(c2-c1) / np.sqrt(a*a+b*b)
        
    print("distance=", distance, a1,a2,b1,b2,c1,c2)
    return distance



def gapBetweenLines(l1,l2):
    #np.seterr(over='ignore')
    
    x12    = np.float(l1[2])
    x10    = np.float(l1[0])
    x22    = np.float(l2[2])
    x20    = np.float(l2[0])
    
    dx1a   = np.max((x12,x10))
    dx2a   = np.min((x22,x20))
    dx1b   = np.min((x12,x10))
    dx2b   = np.max((x22,x20))
    dxmin  = np.min((np.abs(dx1a-dx2a), np.abs(dx1b-dx2b)))

    y13    = np.float(l1[3])
    y11    = np.float(l1[1])
    y23    = np.float(l2[3])
    y21    = np.float(l2[1])
    
    dy1a   = np.max((y13,y11))
    dy2a   = np.min((y23,y21))
    dy1b   = np.min((y13,y11))
    dy2b   = np.max((y23,y21))
    dymin  = np.min((np.abs(dy1a-dy2a), np.abs(dy1b-dy2b)))
    
    return np.sqrt(dxmin*dxmin+dymin*dymin)
    


def isParalell(l1,l2):
    #np.seterr(over='ignore')
    
    slope1 = np.float(0)
    slope2 = np.float(0)
    x12    = np.float(l1[2])
    x10    = np.float(l1[0])
    x22    = np.float(l2[2])
    x20    = np.float(l2[0])
    dx1    = np.float(x12-x10)
    dx2    = np.float(x22-x20)
    
    y13    = np.float(l1[3])
    y11    = np.float(l1[1])
    y23    = np.float(l2[3])
    y21    = np.float(l2[1])
    
    #print ("dx=", dx1, dx2)
    # vertically paralell
    paralell = (np.abs(dx1) < 1.0) and (np.abs(dx2) < 1.0)
    # print ("isParalell(Vert)=",paralell, l1, l2)
    if (not paralell):
        
        if np.abs(dx1)<2: 
            slope1 = np.sign(dx1)*1.0e3
        else:
            slope1 = np.divide((y13-y11), dx1)

        if np.abs(dx2)<2:
            slope2 = np.sign(dx2)*1.0e3
        else:
            slope2 = np.divide((y23-y21), dx2)
            
        paralell = (np.abs(slope2-slope1) < 0.2)

    return paralell


def isLonger(l1,l2):
    
    len1       = np.sqrt((l1[3]-l1[1])*(l1[3]-l1[1]) + (l1[2]-l1[0])*(l1[2]-l1[0]))
    len2       = np.sqrt((l2[3]-l2[1])*(l2[3]-l2[1]) + (l2[2]-l2[0])*(l2[2]-l2[0]))
    l1IsLonger = False
    l2IsLonger = False
    
    if len1 > 2*len2:
        l1IsLonger = True
    elif len2 > 2*len1:
        l2IsLonger = True
        
    return l1IsLonger, l2IsLonger


def plot (imageOriginal, lineFit, lineGood, roiVertice, verticeI):  
  
    cv2.polylines(imageOriginal, roiVertice, True, [0,255,0], 1)

    # print(VerticeI)
    # stop()
    #print('lineBetter=', lineBetter)  
    # draw all the lines found
    try:
        for coords in lineGood:
            #print('line coords: ', coords[0], coords[1]), (coords[2], coords[3])
            try:
                #coords = transform(coords, 0.5)
                
                cv2.line(imageOriginal, (coords[0], coords[1]), (coords[2], coords[3]), [255,105,180], 3)
                # transpose
                # cv2.line(imageOriginal, (coords[1], coords[0]), (coords[3], coords[2]), [255,0,0], 2)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno, str(e))
 
#        rrr =        [lineBetter.reshape(4,2)]
#        cv2.polylines(imageOriginal, rrr, False, [0,0,255], 1)

        for coords in lineFit:
            #print('line coords: ', coords[0], coords[1]), (coords[2], coords[3])
            try:
                #coords = transform(coords, 0.5)
                
                numOfCol  = len(coords)
                posOfHalf = np.uint(numOfCol/2)
                x         = coords[0:posOfHalf]
                y         = coords[posOfHalf:numOfCol]
                x = x.astype(int)
                y = y.astype(int)
                arrayTmp  = np.array([x,y]).transpose()
                # arrayTmp  = np.uint16(arrayTmp)
                cv2.polylines(imageOriginal, [arrayTmp], False, [255,0,0], 2)

                # cv2.line(imageOriginal, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 2)
                # transpose
                # cv2.line(imageOriginal, (coords[1], coords[0]), (coords[3], coords[2]), [255,0,0], 2)
            except Exception as e:
                                
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno,str(e))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        pass

    # return m1, m2
    
def transform(coords, scale):
    # x1, y1
    coords[0] = coords[0]*scale
    coords[1] = coords[1]*scale
    # x2,y2
    coords[2] = coords[2]*scale
    coords[3] = coords[3]*scale
    return coords

#
#    get line by intensity of a horizontal line
#    
def lineByIntensity(image):
    
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab) # 1: +
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #  
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2Luv) #  
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ) #  
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) #  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    
    pdx = pd.partialX(image[:,:,1], 3)
    pdy = pd.partialY(image[:,:,1], 3)
    ds  = np.sqrt(pdx*pdx + pdy*pdy)
    #ds  = cv2.GaussianBlur(ds, (7,7), 0)
    
    pdx = pd.partialX(image[:,:,2], 3)
    pdy = pd.partialY(image[:,:,2], 3)
    dv  = np.sqrt(pdx*pdx + pdy*pdy)
    #dv  = cv2.GaussianBlur(dv, (7,7), 0)
    dsv    = np.sqrt(ds*ds + dv*dv)
    dsMax  = np.amax(ds)    
    dvMax  = np.amax(dv)    
    dsvMax = np.amax(dsv)

    
    ds  = np.array(ds*255/dsMax,   dtype=np.uint8)
    dv  = np.array(dv*255/dvMax,   dtype=np.uint8)
    dsv = np.array(dsv*255/dsvMax, dtype=np.uint8)
    
    #dsv  = cv2.GaussianBlur(dsv, (7,7), 0)
    print('ervervr ', image.shape,ds.shape, dv.shape, dsv.shape)
    indexEndx   = np.uint32(len(image[1,:,1]))
    indexStartx = np.uint32(0*indexEndx/2)
    indexEndy   = np.uint32(len(image[:,1,1]))
    y   = np.ones(indexEndx-indexStartx)
    x   = np.arange(indexStartx,indexEndx)
    
#    ax  = fig0.add_subplot(111)
#    plt.imshow(image)
#    plt.show()
#    stop()    
    fig0  = plt.figure(figsize=plt.figaspect(1)*1.5)
    ax0   = fig0.add_subplot(231, projection='3d')
    ax1   = fig0.add_subplot(232, projection='3d')
    ax2   = fig0.add_subplot(233, projection='3d') 
    ax3   = fig0.add_subplot(234, projection='3d') 
    ax4   = fig0.add_subplot(235, projection='3d') 
    ax5   = fig0.add_subplot(236, projection='3d')
    
    fig1  = plt.figure()
    ax11  = fig1.add_subplot(231)
    ax12  = fig1.add_subplot(232)
    ax13  = fig1.add_subplot(233)
    ax14  = fig1.add_subplot(234)
    ax15  = fig1.add_subplot(235)
    ax16  = fig1.add_subplot(236)
    
    
    print(x.shape,y.shape, image.shape )
     
    # do right
    for i in range(0, indexEndy, 10):
        # print('i=',i)
        line0, = ax0.plot(np.flipud(x), y*i, image[i,indexStartx:,0], '-', linewidth=1, color='r', label='first')
        line1, = ax1.plot(np.flipud(x), y*i, image[i,indexStartx:,1], '-', linewidth=1, color='g', label='second')
        line2, = ax2.plot(np.flipud(x), y*i, image[i,indexStartx:,2], '-', linewidth=1, color='b', label='third')
      
        line3, = ax3.plot(np.flipud(x), y*i, ds[i,indexStartx:],  '-', linewidth=1, color='r', label='dS')
        line4, = ax4.plot(np.flipud(x), y*i, dv[i,indexStartx:],  '-', linewidth=1, color='g', label='dV')
        line5, = ax5.plot(np.flipud(x), y*i, dsv[i,indexStartx:], '-', linewidth=1, color='b', label='dSV')


    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    
    ax0.auto_scale_xyz([0, 801], [0, 581], [0, 255*8])
    ax1.auto_scale_xyz([0, 801], [0, 581], [0, 255*8])
    ax2.auto_scale_xyz([0, 801], [0, 581], [0, 255*8])
    
    ax3.auto_scale_xyz([0, 801], [0, 581], [0, 25*8])
    ax4.auto_scale_xyz([0, 801], [0, 581], [0, 25*8])
    ax5.auto_scale_xyz([0, 801], [0, 581], [0, 25*8])
    
    ax0.view_init(60,70)
    ax1.view_init(60,70)
    ax2.view_init(60,70)

    ax3.view_init(60,70)
    ax4.view_init(60,70)
    ax5.view_init(60,70)
    

    ds = cv2.GaussianBlur(ds, (7,7), 0)
    plt.axes(ax14)
    plt.imshow(ds, cmap='gray')
    
    dv = cv2.GaussianBlur(dv, (7,7), 0)
    plt.axes(ax15)
    plt.imshow(dv, cmap='gray')
    
    dsv = cv2.GaussianBlur(dsv, (7,7), 0)
    plt.axes(ax16)
    plt.imshow(dsv, cmap='gray')


    image1 =   image[:,:,0]
    plt.axes(ax11)
    plt.imshow(image1, cmap='gray')
     
    image1 =   image[:,:,1]
    plt.axes(ax12)
    plt.imshow(image1, cmap='gray')
    
    image1 =   image[:,:,2]
    plt.axes(ax13)
    plt.imshow(image1, cmap='gray')
    
         
    #print('image size=', image0.shape, image1.shape, image2.shape, image3.shape, )
    plt.show()
    
    stop()

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
