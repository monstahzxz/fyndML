import os
import cv2
import urllib.request
import numpy as np

def makeDataSet(full_path, save_path):
	exclude = 1
	dims = (200,200)
	outFile = open(save_path,'w',encoding='utf-8')
	with open(full_path,'r',encoding='utf-8') as inFile:
		for line in inFile:
			columns = line.split(',')	
			idx, urls, classx = columns[0], columns[1:6], columns[6]

			images = ''
			for url in urls:
				if not url:
					continue

				with urllib.request.urlopen(url) as res:
					image = np.asarray(bytearray(res.read()), dtype='uint8')
					image = cv2.imdecode(image, cv2.IMREAD_COLOR)
					img_resized = cv2.resize(image, dims)
					img_resized_flattened = img_resized.flatten()
					str_img_resized = [str(pix) for pix in img_resized_flattened]
					images += " ".join(str_img_resized) + ','	

					#print(images)
			
			outFile.write(idx + ',' + images + classx)
			print('Image set ' + str(exclude) + ' downloaded...')
			exclude += 1
			
	outFile.close()		


if __name__ == '__main__':
	data_path = './Dataset/fyndURL.csv'
	save_path = './Dataset/fyndData.csv'
	makeDataSet(data_path, save_path)