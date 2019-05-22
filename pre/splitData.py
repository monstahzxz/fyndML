import cv2
import numpy as np
import matplotlib.pyplot as plt



f = open('./Dataset/fyndData.csv','r',encoding='utf-8')
out = open('./Dataset/fyndDataSplit.csv','w',encoding='utf-8')
i = 0

for line in f:
	i = i + 1
	if not line:
		continue
	columns = line.strip().lower().split(',')
	idx, image_cols, classx = columns[0], columns[1:-1], columns[-1]
	
	for image in image_cols:
		out.write(idx + ',' + image + ',' + classx + '\n')

	print('Image set ' + str(i) + ' over')	

f.close()
out.close()	