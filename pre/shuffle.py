import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


#def model(X, Y):






if __name__ == '__main__':
	dataset_path = os.getcwd() + '/Dataset/fyndDataSplit.csv'
	dims = (200,200)

	f = open('./Dataset/fyndDataSplit.csv','r',encoding='utf-8')
	out = open('./Dataset/ShuffledfyndDataSplit.csv','w',encoding='utf-8')
	i = 0
	X = []
	for line in f:
		i = i + 1
		if not line:
			continue
		
		columns = line.strip().lower().split(',')
		image_col = columns[1]
		image_col = [int(x) for x in image_col.split()]
		image = np.array(image_col,dtype='uint8')
		image = image.reshape((200,200,-1))
		img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		test = (columns[0],img_gray,columns[2])
		X.append(test)
		if i % 250 == 0:
			print(str(i) + ' done')

	np.random.shuffle(X)

	for i in range(len(X)):
		img_resized_flattened = X[i][1].flatten()
		str_img_resized = [str(pix) for pix in img_resized_flattened]
		images = " ".join(str_img_resized) + ','
		out.write(X[i][0] + ',' + images + X[i][2] + '\n')

	print('Done.')  
		