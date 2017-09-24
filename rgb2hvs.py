import numpy as np
import cv2
sample = [(
	[150,98,52], 
	[149,91,41], 
	[150,95,46],
	[125,82,45],
	[142,91,46],
	[137,92,53],
	[124,82,46],
         )]
  
sample = np.array(sample, dtype=np.uint8)
hsv    = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)
print ('hsv dirty white: \n', hsv[0])
#print ('hsv min,max: \n', np.max((hsv[0][:,0],hsv[0][:,1],hsv[0][:,2])))
print ('hsv min: \n', np.min((hsv[0]), 0))
print ('hsv max: \n', np.max((hsv[0]), 0))

#print ('ercr:',boundaryRGB[:,-1:])

  # yellow in tunnel light and dark
# [66,58,41], [51,45,31]
 
# # dirty yellow
# sample = [([253,253,247],[253,253,247])]
# sample = np.uint8(sample)
# hsv    = cv2.cvtColor(sample,cv2.COLOR_RGB2HSV)
# print ('hsv qqdirty yellow: \n', hsv[0])


# # yellow in tunnel but not in dark
# sample = [([66,58,41], [51,45,31])]
# sample = np.uint8(sample)
# hsv    = cv2.cvtColor(sample,cv2.COLOR_RGB2HSV)
# print ('hsv yellow in tunnel: \n', hsv[0])


# # white dashed line under the sun
# sample = [([244,239,218], [187,181,157])]
# sample = np.uint8(sample)
# hsv    = cv2.cvtColor(sample,cv2.COLOR_RGB2HSV)
# print ('hsv white dashed line: \n', hsv[0])

# # white dashed line under the sun
# sample = [([251,251,249], [249,246,241])]
# sample = np.uint8(sample)
# hsv    = cv2.cvtColor(sample,cv2.COLOR_RGB2HSV)
# print ('hsv very white dashed line: \n', hsv[0])
