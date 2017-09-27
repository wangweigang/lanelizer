#------------------------------------------
# Starting lanelizer (adapted from sendex)
#------------------------------------------
import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import lane 
import imageProc as imgProc

import edgeProc as edProc

import os, sys

indexStart   = 1
imageCounter = 0

#
# check data previously saved: if none, start from index 1; if exists, save data further with continuing index
#
while True:

    fileName = 'E:/temp/pygta5/dataCollection/training_data-{}.npy'.format(indexStart)
    if os.path.isfile(fileName):
        print(fileName)
        print('File exists, moving along',indexStart)
        indexStart += 1
    else:
        print('File does not exist, starting fresh!',indexStart)

        break

w  = [1,0,0,0,0,0,0,0,0]
s  = [0,1,0,0,0,0,0,0,0]
a  = [0,0,1,0,0,0,0,0,0]
d  = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]


def keys2output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output


def frameProc (fileName, training_data, roiVertice, verticeI, indexStart, imageCounter):
    #
    # image/training data collection
    #
    paused = False
    print('Start image collection ... ...')

    while(True):
        imageCounter += 1
        if not paused:
            if imageCounter % 1 == 0:
                # grab the image at the top-left
                timeStart = time.time()
                
                # grab a frame at the top left of the screen 800x600 of Osiris
                # image = grab_screen(region=(0,36,800,616))
                # grab a frame at the top left of the screen 1024x678 of Osiris
                image = grab_screen(region=(190,240, 190+835,240+310))
                
                timeGrab = time.time()
                
                # store original image for display
                imageOriginal  = image
#                imageOriginal1 = np.copy(image)
                 
                # process image 
                image = imgProc.imageProc(image, roiVertice, verticeI)
                timeImgProcess = time.time()
                
                # find edges
                image = edProc.edgeProc(image)
                timeEdgeProcess = time.time()
                
                # find lines
                lines = lane.lineFind (imageOriginal, image)
                timeLineFind = time.time() 
                # cv2.polylines(imageOriginal, verticeI, True, [0,0,255], 1)
                
                try:
                    if len(lines)>=1:
                        laneGood = lane.laneFind(lines, roiVertice)
                        # draw lanes over original image
                        # print("laneGood: ", laneGood.shape)
                        # print("laneBetter[0]: ", laneBetter.shape)
                        
                        # grouping lines
                        laneFit = lane.lineGroup(laneGood)
                        
                        lane.plot (imageOriginal, laneFit, laneGood, roiVertice, verticeI)
#                        lane.plot (imageOriginal1, lines, roiVertice, verticeI)
                except Exception as e:
                    # print("line=", lines) 
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno,str(e))
                    pass
                
                #stop()
                timeLaneFind = time.time()

                cv2.imshow('Original', imageOriginal)
                timeLanePlot = time.time()
                
                # lane.plot (imageOriginal1, laneBetter, roiVertice)
                # cv2.imshow('Original (better)', imageOriginal1 )

                keys   = key_check()
                output = keys2output(keys)

                training_data.append([image, output])

                print("Grabing fps: %3d "      % (1 / (timeGrab-timeStart) if timeGrab != timeStart else 0.00001), 
                      " Image process: %3d ms" % (1000*(timeImgProcess-timeGrab)),
                      " Edge finding: %2d ms"  % (1000*(timeEdgeProcess-timeImgProcess)),
                      " Line finding: %2d ms"  % (1000*(timeLineFind-timeEdgeProcess)),
                      " Lane finding: %2d ms"  % (1000*(timeLaneFind-timeLineFind)),
#                          "Lane plotting %3d ms" % (1000*(timeLanePlot-timeLaneFind)),
                      " Total fps: %2d"       % (1 / (timeLanePlot-timeStart)) 
                      )
                   
                    # print('fps: %2d frames' % (1 / (current_time-timeStart)), "good better: ", len(laneGood), len(laneBetter))
                    # print('fps: {0}'.format(1 / (current_time-timeStart)))

                # image = cv2.transpose(image)
                # cv2.imshow('Edges', cv2.resize(image, (280,216)))   # for 800x600
                 
                cv2.imshow('Edges', image) # cv2.resize(image, (417,155)))

                # cv2.imshow('Original Image', np.array(imageOriginal))
                
                # press q to quit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                   cv2.destroyAllWindows()
                   break
                   
                if len(training_data) % 100 == 0:
                    print('len of training_data: ',len(training_data))

                    if len(training_data) == 500:
                        ### np.save(fileName,training_data)
                        print('SAVED')
                        training_data = []
                        indexStart += 1
                        fileName = 'E:/temp/pygta5/dataCollection/training_data-{}.npy'.format(indexStart)

                imageCounter = 0
    # return

#
# stop by click close window  
#        intenMax = image[:,i,1]
def stop():
    cv2.waitKey(0)
    # input('Any key to stop ... ...')
    cv2.destroyAllWindows()
    sys.exit()    

def main(fileName, indexStart, imageCounter):
    indexStart = indexStart
    training_data = []
    # for i in list(range(4))[::-1]:
        # print(i+1)
        # time.sleep(1)

    # can have several vertice sets

    # front window for Osiris 800x600
    #                          0   1     2   3     4   5     6   7     8   9
    # roiVertice = np.array([[140,176],[140,408],[469,360],[797,408],[797,176]], np.int32)
    #                          0   1     2   3     4   5     6   7     8   9
    # roiVertice = np.array([[140,176],[140,408],[469,408],[797,408],[797,176]], np.int32)
    # counter clockwise
    #                          0   1     2   3     4   5     6   7 
    # roiVertice = np.array([[140,176],[140,408],[797,408],[797,176]], np.int32)

    # front window for Osiris 1024x678
    # roiVertice = np.array([[270,176+22],[920,176+22],[1024,408+99],[190,408+99]], np.int32)
    # roiVertice = np.array([[270,176+22],[920,176+22],[1
     
     
    roiVertice = np.array([[32,0],[835-32,0],[835,310],[0,310]], np.int32)
    # 190,240, 190+835,240+310)
    # clockwise and compensate the cv2's poly distortion
    # roiVertice = np.array([[140,148],[788,164],[804,408],[156,408]], np.int32)
    
    # for light intensity measurement
    vertice1 = np.array([[390,200],[660,260],[120,260]], np.int32)
 
    frameProc (fileName, training_data, [roiVertice],[vertice1], indexStart, imageCounter)



main(fileName, indexStart, imageCounter)
