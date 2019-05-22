import os
import cv2
import urllib.request
import numpy as np

def makeDataSet(full_path, save_path):
	exclude = 1
	outFile = open(save_path,'a+',encoding='utf-8')
	with open(full_path,'r',encoding='utf-8') as inFile:
		for line in inFile:
			if exclude < 2:
				exclude += 1
				continue

			#if exclude == 104:
			#	break
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
			print('Image set ' + str(exclude - 1) + ' done...')
			exclude += 1
			

	outFile.close()		

			

			





if __name__ == '__main__':
	data_path = './Dataset/fyndURL.csv'
	save_path = './Dataset/fyndData.csv'
	dims = (200,200)
	makeDataSet(data_path, save_path)


'''f = open(data_path + 'fyndURL.csv','r')
count = 1

for line in f:
	if count == 1:
		count += 1
		continue
	url1, url2, url3, url4, url5 = line.split(',')[1:6]
	with urllib.request.urlopen(url1) as res:
		image = np.asarray(bytearray(res.read()), dtype='uint8')
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)

		cv2.imshow('yep',image)
		img_resized = cv2.resize(image, (200,200))
		cv2.imwrite('img.jpg',img_resized)
		#print(image.shape)
		cv2.waitKey(0)
	break'''