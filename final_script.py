import sys
sys.path.append('./src/')
sys.path.append('./pre/')
sys.path.append('./model_def/')
from pre import makeDataSet
from gray4final import makeGray
#from bigcldnn import placeholders, conv_layers, conv_re_layers, tie_input_timesteps, lstm_layers, fc_layers, model, getData
from predbigcldnn import model, getData
import json

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('Pass filename as argument 1 and downloaded(1 or 0) as argument 2')
	else:
		dataset_path = sys.argv[1]
		downloaded = 1 if sys.argv[2] == '1' else 0

		with open('./model_def/bigcldnn.json','r') as f:
			defs = json.load(f)

		save_path = defs['model_path']
		def_path = defs['def_path']

		dims = defs['dims'] 
		output_classes = defs['output_classes']

		ground_truth = {
			'zipper' : 0,
			'backstrap' : 1,
			'slip_on' : 2,
			'lace_up' : 3,
			'buckle' : 4,
			'hook&look' : 5,
		}

		conv1_fmaps = defs['conv1_fmaps'] 
		conv1_ksize = defs['conv1_ksize'] 
		conv1_stride = defs['conv1_stride']
		conv1_pad = defs['conv1_pad'] 

		conv2_fmaps = defs['conv2_fmaps'] 
		conv2_ksize = defs['conv2_ksize'] 
		conv2_stride = defs['conv2_stride'] 
		conv2_pad = defs['conv2_pad']

		conv3_fmaps = defs['conv3_fmaps'] 
		conv3_ksize = defs['conv3_ksize'] 
		conv3_stride = defs['conv3_stride']
		conv3_pad = defs['conv3_pad'] 

		conv4_fmaps = defs['conv4_fmaps'] 
		conv4_ksize = defs['conv4_ksize'] 
		conv4_stride = defs['conv4_stride'] 
		conv4_pad = defs['conv4_pad']

		fc_dims = defs['fc_dims']

		fc_size1 = defs['fc_size1']
		fc_size2 = defs['fc_size2'] 

		if downloaded == 0:
			makeDataSet(dataset_path,'pre.csv')
			makeGray('pre.csv','gray.csv')
		else:
			makeGray(dataset_path,'gray.csv')
		X, Y, idx = getData('gray.csv',defs)
		model(X, Y, idx, save_path, def_path)