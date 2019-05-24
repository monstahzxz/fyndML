#from cldnn import placeholders, conv_layers, conv_re_layers, tie_input_timesteps, lstm_layers, fc_layers
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import math
import json


def placeholders(defs):
	dims = defs['dims'] 
	output_classes = defs['output_classes']

	with tf.variable_scope('placeholders'):
		x = []
		for i in range(5):
			x.append(tf.placeholder(tf.float32,[None,dims,dims,1],name='x' + str(i + 1)))
		
		y = tf.placeholder(tf.float32,[None,output_classes],name='y')

	return x,y

def conv_layers(x,defs):
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

	with tf.variable_scope('convs'):
		conv1 = tf.layers.conv2d(x, filters=conv1_fmaps, kernel_size = conv1_ksize, strides = conv1_stride, padding=conv1_pad, activation = tf.nn.relu, name="conv1")
		pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		conv2 = tf.layers.conv2d(pool1, filters=conv2_fmaps, kernel_size = conv2_ksize, strides = conv2_stride, padding=conv2_pad, activation = tf.nn.relu, name="conv2")
		pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		conv3 = tf.layers.conv2d(pool2, filters=conv3_fmaps, kernel_size = conv3_ksize, strides = conv3_stride, padding=conv3_pad, activation = tf.nn.relu, name="conv3")
		pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		conv4 = tf.layers.conv2d(pool3, filters=conv4_fmaps, kernel_size = conv4_ksize, strides = conv4_stride, padding=conv4_pad, activation = tf.nn.relu, name="conv4")
		pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		x_fc = tf.reshape(pool4, shape = [-1, 484])

		tf.summary.histogram('conv1',conv1)
		tf.summary.histogram('pool1',pool1)
		tf.summary.histogram('conv2',conv2)
		tf.summary.histogram('pool2',pool2)
		tf.summary.histogram('conv3',conv3)
		tf.summary.histogram('pool3',pool3)
		tf.summary.histogram('conv4',conv4)
		tf.summary.histogram('pool4',pool4)
	
	return x_fc

def conv_re_layers(x,defs):
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

	with tf.variable_scope('convs', reuse=True):
		conv1 = tf.layers.conv2d(x, filters=conv1_fmaps, kernel_size = conv1_ksize, strides = conv1_stride, padding=conv1_pad, activation = tf.nn.relu, name="conv1")
		pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		conv2 = tf.layers.conv2d(pool1, filters=conv2_fmaps, kernel_size = conv2_ksize, strides = conv2_stride, padding=conv2_pad, activation = tf.nn.relu, name="conv2")
		pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		conv3 = tf.layers.conv2d(pool2, filters=conv3_fmaps, kernel_size = conv3_ksize, strides = conv3_stride, padding=conv3_pad, activation = tf.nn.relu, name="conv3")
		pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		conv4 = tf.layers.conv2d(pool3, filters=conv4_fmaps, kernel_size = conv4_ksize, strides = conv4_stride, padding=conv4_pad, activation = tf.nn.relu, name="conv4")
		pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		x_fc = tf.reshape(pool4, shape = [-1, 484])

	
	return x_fc


def tie_input_timesteps(xs):
	return tf.stack([xi for xi in xs])


def lstm_layers(x_fcs_input):
	with tf.variable_scope('lstm1'):
		lstm_layer = tf.contrib.rnn.LSTMCell(128)
		outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_layer, lstm_layer, x_fcs_input, dtype=tf.float32)

		tf.summary.histogram('lstm outputs', outputs)

	return outputs



def fc_layers(x_fc,defs):
	fc_dims = defs['fc_dims']

	fc_size1 = defs['fc_size1']
	fc_size2 = defs['fc_size2'] 

	with tf.variable_scope('fc'):
		w1 = tf.get_variable("w1",[fc_size1,fc_size2],initializer = tf.contrib.layers.xavier_initializer())
		b1 = tf.get_variable("b1", [fc_size2], initializer = tf.zeros_initializer())

		tf.summary.histogram('w1', w1)
		tf.summary.histogram('b1', b1)

		z1 = tf.add(tf.matmul(x_fc,w1),b1)


	return z1


def model(X, Y, idx, save_path='../Models/bigcldnn/bigcldnn.ckpt', def_path='../model_def/bigcldnn.json'):
	with open(def_path,'r') as f:
		defs = json.load(f)
	ground_truth = defs['ground_truth']

	out = open('output.csv','w',encoding='utf-8')
	x, y = placeholders(defs)
	x_fcs = []
	x_fcs.append(conv_layers(x[0],defs))
	for i_conv in range(4):
		x_fcs.append(conv_re_layers(x[i_conv + 1],defs))

	with tf.variable_scope('prestack'):
		x_fcs_tied = tie_input_timesteps(x_fcs)		
		x_fcs_tied_proper = tf.transpose(x_fcs_tied, perm=[1,0,2])
		x_fcs_input = tf.unstack(x_fcs_tied_proper, 5, 1)
		
	with tf.variable_scope('lstm'):	
		x_fcs_output = tf.transpose(lstm_layers(x_fcs_input), perm=[1,0,2])
		x_fcs_output_reshaped = tf.reshape(x_fcs_output,[-1,5*256])

	z1 = fc_layers(x_fcs_output_reshaped,defs)
	
	with tf.variable_scope('cost'):
		cost = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z1,labels=y,name='loss'))

	with tf.variable_scope('accuracy'):	
		equality = tf.equal(tf.argmax(z1, 1), tf.argmax(y, 1))
		acc = tf.reduce_mean(tf.cast(equality, tf.float32))

	with tf.variable_scope('optimiser'):	
		optimiser = tf.train.AdamOptimizer().minimize(cost)

	#init = tf.global_variables_initializer()
	sess = tf.Session()

	##saver for checkpointing
	saver = tf.train.Saver()

	##initialise variables	
	#sess.run(init)
	
	##Restore cldnn
	saver.restore(sess,save_path)

	e_z1, e_cost = sess.run([z1, cost], feed_dict = {
				x[0]:X[:,0,:,:],
				x[1]:X[:,1,:,:],
				x[2]:X[:,2,:,:],
				x[3]:X[:,3,:,:],
				x[4]:X[:,4,:,:],
				y:Y
				})
	#print(e_z2)
	pred = np.argmax(e_z1, axis = 1)
	label = np.argmax(Y, axis = 1)

	correct = 0
	#print(pred)
	#print(label)
	for i in range(len(pred)):
		for key in ground_truth:
			if ground_truth[key] == pred[i]:
				classx = key
				break
		out.write(str(idx[i]) + ',' + str(classx))
		if pred[i] == label[i]:
			correct += 1

	print('Accuracy = ' + str(correct/len(pred) * 100) + '%')	
	print('Error = ' + str(e_cost))		
	'''for key in ground_truth:
		if ground_truth[key] == pred:
			print('Pred: ' + key)
			break
	for key in ground_truth:
		if ground_truth[key] == label:
			print('Label: ' + key)
			break'''

	conf_mat = np.zeros((6,6))

	print('-----------')
	print('Confusion Matrix')
	print('-----------')
	for i in range(len(pred)):
		conf_mat[label[i]][pred[i]] += 1

	print(conf_mat)
					



def getData(dataset_path,defs):
	X = []
	Y = []
	i = 0
	idx = []
	ground_truth = defs['ground_truth']
	output_classes = defs['output_classes']

	inFile = open(dataset_path,'r',encoding='utf-8')
	'''for it in range(1900):
		inFile.readline()'''

	for line in inFile:
		i += 1
		X.append([])
		columns = line.strip().lower().split(',')
		id_, images, classx = columns[0], columns[1:-1], columns[-1]
		for image in images:
			image = [int(x) for x in image.split()]
			image = np.array(image,dtype='uint8').reshape(200,200,-1)
			X[i - 1].append(image)
		if len(X[i - 1]) < 5:
			for ith in range(5 - len(X[i - 1])):
				X[i - 1].append(np.zeros((image.shape)))	
		temp = [0] * output_classes
		temp[ground_truth[classx]] = 1
		Y.append(temp)
		idx.append(id_)

		if i % 250 == 0:
			print('Read ex: ' + str(i))
					

	inFile.close()	

	X = np.asarray(X)

	return X, Y, idx	


if __name__ == '__main__':
	
	with open('../model_def/bigcldnn.json','r') as f:
		defs = json.load(f)


	dataset_path = defs['test_dataset_path']
	output_classes = defs['output_classes']
	epochs = defs['epochs'] 
	batch_size = defs['batch_size']
	dataset_size = defs['dataset_size'] 
	batches = defs['batches'] 
	
	ground_truth = {
		'zipper' : 0,
		'backstrap' : 1,
		'slip_on' : 2,
		'lace_up' : 3,
		'buckle' : 4,
		'hook&look' : 5,
	} 

	X, Y, idx = getData(dataset_path,defs)
	
	print('Read data successfully!')

	model_save_path = defs['model_save_path']
	visual_save_path = defs['visual_save_path']

	model(X,Y, idx)