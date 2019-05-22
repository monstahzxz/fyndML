import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


#def model(X, Y):






if __name__ == '__main__':
	dims = (200,200)

	f = open('./mined_dataset/2/test_data_dl_final.csv','r',encoding='utf-8')
	out = open('./mined_dataset/2/final.csv','w',encoding='utf-8')
	i = 0
	X = []
	for line in f:
		i = i + 1
		if not line:
			continue
		
		columns = line.strip().lower().split(',')
		
		if columns[-1] == 'buckle':
			print('buckle.. skipping')
			continue

		image_cols = columns[1:-1]
		img_grays = []
		for image_col in image_cols:
			image_col1 = [int(x) for x in image_col.split()]
			image = np.array(image_col1,dtype='uint8')
			image = image.reshape((200,200,-1))
			img_grays.append(image)
		test = (columns[0],img_grays,columns[-1])
		X.append(test)
		if i % 100 == 0:
			print(str(i) + ' done')	


	f = open('./Dataset/fyndDataGray.csv','r',encoding='utf-8')
	i = 0
	#X = []
	for line in f:
		i = i + 1
		if not line:
			continue

		columns = line.strip().lower().split(',')
		image_cols = columns[1:-1]
		img_grays = []
		for image_col in image_cols:
			image_col1 = [int(x) for x in image_col.split()]
			image = np.array(image_col1,dtype='uint8')
			image = image.reshape((200,200,-1))
			img_grays.append(image)
		test = (columns[0],img_grays,columns[-1])
		X.append(test)

		if columns[-1] == 'buckle':
			print('buckle... replicating x4')
			X.append(test)
			X.append(test)
			X.append(test)

		if i % 100 == 0:
			print(str(i) + ' done')

		if i % 1900 == 0:
			break
				

	np.random.shuffle(X)


	for i in range(len(X)):
		out.write(X[i][0] + ',')
		for img in X[i][1]:
			img_resized_flattened = img.flatten()
			str_img_resized = [str(pix) for pix in img_resized_flattened]
			images = " ".join(str_img_resized) + ','
			out.write(images)
		out.write(X[i][-1] + '\n')

		if i % 100 == 0:
			print(str(i) + ' done')	

	print('Done.')  
		