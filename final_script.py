import sys
sys.path.append('./src/')
sys.path.append('./pre/')
sys.path.append('./model_def/')
from pre import makeDataSet
from gray4final import makeGray
from predbigcldnn import model, getData
import json

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('Pass filename as argument 1 and downloaded(1 or 0) as argument 2')
	else:
		# Setting dataset path and type of CSV file as obtained from command line
		dataset_path = sys.argv[1]
		downloaded = 1 if sys.argv[2] == '1' else 0

		# Data definitions (Hyperparameters, model paths, etc)
		with open('./model_def/bigcldnn.json','r') as f:
			defs = json.load(f)

		save_path = defs['model_path']
		def_path = defs['def_path']

		# For CSVs having URLs
		if downloaded == 0:
			makeDataSet(dataset_path,'pre.csv')
			makeGray('pre.csv','gray.csv')
		# For CSVs having images
		else:
			print('Ensure that you are passing on BGR flattened images, and not RGB (Conversion is made from BGR to grayscale, if RGB is passed, results will be wrong)')
			makeGray(dataset_path,'gray.csv')
		
		# Reading data in-memory
		X, Y, idx = getData('gray.csv',defs)

		# Deploying model to make predictions
		model(X, Y, idx, save_path, def_path)