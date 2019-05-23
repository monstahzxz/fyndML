import cv2
import numpy as np
import matplotlib.pyplot as plt

#a = [0,8,15,49,66,83,90,115,120,125,128,138,141,147,153,172,175,194,196,204,249]

f = open('Dataset/fyndData.csv','r',encoding='utf-8')
i = 0

#for it in range(1900):
#	i += 1
#	f.readline()

for line in f:
	if not line:
		continue

	i += 1	
	columns = line.strip().lower().split(',')
	image_col = columns[1]
	image_col = [int(x) for x in image_col.split()]
	#print(image.reshape((200,200,-1)).shape)
	image = np.array(image_col,dtype='uint8')
	image = image.reshape((200,200,-1))
	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #UN-COMMENT THIS LINE FOR BGR IMAGES (put img_gray in imshow also)
	#if i in a:
	cv2.imshow('ye',image)
	print(columns[-1])
	print(i)
	cv2.waitKey(0)
	cv2.imshow('df',img_gray)
	cv2.waitKey(0)


f.close()

#1550
#1774
#1966
#2098