import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


#def model(X, Y):






if __name__ == '__main__':
	dataset_path = os.getcwd() + '/Dataset/fyndData.csv'
	dims = (200,200)

	f = open(dataset_path,'r',encoding='utf-8')
	out = open('./Dataset/fyndDataColor.csv','w',encoding='utf-8')
	i = 0
	X = []
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
			img_grays.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) #For gray
			#img_grays.append(image) #For color
		test = (columns[0],img_grays,columns[-1])
		X.append(test)
		if i % 100 == 0:
			print(str(i) + ' done')

	np.random.shuffle(X)

	for i in range(len(X)):
		out.write(X[i][0] + ',')
		for img in X[i][1]:
			img_resized_flattened = img.flatten()
			str_img_resized = [str(pix) for pix in img_resized_flattened]
			images = " ".join(str_img_resized) + ','
			out.write(images)
		out.write(X[i][-1] + '\n')	

	print('Done.')  
		