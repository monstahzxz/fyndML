import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import math
import json


def placeholders():
	with tf.variable_scope('placeholders'):
		x = []
		for i in range(5):
			x.append(tf.placeholder(tf.float32,[None,dims,dims,1],name='x' + str(i + 1)))
		
		y = tf.placeholder(tf.float32,[None,output_classes],name='y')

	return x,y

def conv_layers(x):
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

def conv_re_layers(x):
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


def fc_layers(x_fc):
	with tf.variable_scope('fc'):
		w1 = tf.get_variable("w1",[fc_size1,fc_size2],initializer = tf.contrib.layers.xavier_initializer())
		b1 = tf.get_variable("b1", [fc_size2], initializer = tf.zeros_initializer())


		tf.summary.histogram('w1', w1)
		tf.summary.histogram('b1', b1)


		z1 = tf.add(tf.matmul(x_fc,w1),b1)

	return z1

	
def model(X, Y, test_X, test_Y):
	maxAcc, maxAccSess = (0,(0,0))
	x, y = placeholders()
	x_fcs = []
	x_fcs.append(conv_layers(x[0]))
	for i_conv in range(4):
		x_fcs.append(conv_re_layers(x[i_conv + 1]))

	with tf.variable_scope('prestack'):
		x_fcs_tied = tie_input_timesteps(x_fcs)		
		x_fcs_tied_proper = tf.transpose(x_fcs_tied, perm=[1,0,2])
		x_fcs_input = tf.unstack(x_fcs_tied_proper, 5, 1)
		
	with tf.variable_scope('lstm'):	
		x_fcs_output = tf.transpose(lstm_layers(x_fcs_input), perm=[1,0,2])
		x_fcs_output_reshaped = tf.reshape(x_fcs_output,[-1,5*256])

	z1 = fc_layers(x_fcs_output_reshaped)
	
	with tf.variable_scope('cost'):
		cost = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z1,labels=y,name='loss'))

	with tf.variable_scope('accuracy'):	
		equality = tf.equal(tf.argmax(z1, 1), tf.argmax(y, 1))
		acc = tf.reduce_mean(tf.cast(equality, tf.float32))

	with tf.variable_scope('optimiser'):	
		optimiser = tf.train.AdamOptimizer().minimize(cost)

	init = tf.global_variables_initializer()
	sess = tf.Session()

	tf.summary.scalar('Cost', cost)
	tf.summary.scalar('Accuracy', acc)

	##saver for checkpointing
	saver = tf.train.Saver()

	##initialise variables	
	sess.run(init)
	
	##Restore cldnn
	#saver.restore(sess, model_save_path) # D:/fyndML/mined_dataset/2/Models/bigcldnn.ckpt

	##Tensorboard data
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter(visual_save_path) # mined_dataset/2/visual/bigcldnn
	writer.add_graph(sess.graph)
	
	i = 0
	for epoch in range(epochs):
		for batch in range(batches):
			i += 1
			batch_X_1 = X[batch * batch_size:(batch + 1) * batch_size,0,:,:]
			batch_X_2 = X[batch * batch_size:(batch + 1) * batch_size,1,:,:]
			batch_X_3 = X[batch * batch_size:(batch + 1) * batch_size,2,:,:]
			batch_X_4 = X[batch * batch_size:(batch + 1) * batch_size,3,:,:]
			batch_X_5 = X[batch * batch_size:(batch + 1) * batch_size,4,:,:]
			batch_Y = Y[batch * batch_size:(batch + 1) * batch_size]
			_, e_cost = sess.run([optimiser,cost], feed_dict = {
				x[0]:batch_X_1,
				x[1]:batch_X_2,
				x[2]:batch_X_2,
				x[3]:batch_X_4,
				x[4]:batch_X_5,
				y:batch_Y
				})
			#print(e_z3)
			#print(Y)
			print('Cost after epoch ' + str(epoch + 1) + ', batch ' + str(batch + 1) + ' = ' + str(e_cost))

			if i % 2 == 0:
				summ = sess.run(merged_summary, feed_dict = {
				x[0]:batch_X_1,
				x[1]:batch_X_2,
				x[2]:batch_X_2,
				x[3]:batch_X_4,
				x[4]:batch_X_5,
				y:batch_Y
				})
				writer.add_summary(summ, i)

				e_z1 = sess.run(z1, feed_dict = {
				x[0]:test_X[:,0,:,:],
				x[1]:test_X[:,1,:,:],
				x[2]:test_X[:,2,:,:],
				x[3]:test_X[:,3,:,:],
				x[4]:test_X[:,4,:,:],
				})
				
				pred = np.argmax(e_z1, axis = 1)
				label = np.argmax(test_Y, axis = 1)

				correct = 0
				#print(pred)
				#print(label)
				for i in range(len(pred)):
					if pred[i] == label[i]:
						correct += 1
				e_acc = (correct / len(pred)) * 100
				maxAcc, maxAccSess =  (e_acc,(epoch, batch)) if e_acc > maxAcc else (maxAcc, maxAccSess)		
				with open('acc.txt','w',encoding='utf-8') as f:
					f.write(str(maxAcc) + ',' + str(maxAccSess))

			if batch % 10 == 0:
				print('Saving checkpoint...')
				saver.save(sess,model_save_path)
				print('Done saving!')

		saver.save(sess,model_save_path)
		print(maxAcc)
		print(maxAccSess)


def getData(dataset_path):
	X = []
	Y = []
	i = 0

	inFile = open(dataset_path,'r',encoding='utf-8')
	for line in inFile:
		i += 1
		X.append([])
		columns = line.strip().lower().split(',')
		_, images, classx = columns[0], columns[1:-1], columns[-1]
		for image in images:
			image = [int(x) for x in image.split()]
			image = np.array(image, dtype='uint8').reshape(200,200,-1)
			X[i - 1].append(image)
		if len(X[i - 1]) < 5:
			for ith in range(5 - len(X[i - 1])):
				X[i - 1].append(np.zeros((image.shape)))	
		temp = [0] * output_classes
		temp[ground_truth[classx]] = 1
		Y.append(temp)

		if i % 250 == 0:
			print('Read ex: ' + str(i))				
			break
	inFile.close()	

	X = np.asarray(X)

	return X, Y	

def getTestData(dataset_path):
	X = []
	Y = []
	i = 0

	inFile = open(dataset_path,'r',encoding='utf-8')
	for it in range(1900):
		inFile.readline()

	for line in inFile:
		i += 1
		X.append([])
		columns = line.strip().lower().split(',')
		_, images, classx = columns[0], columns[1:-1], columns[-1]
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

		if i % 250 == 0:
			print('Read ex: ' + str(i))					
			
	inFile.close()	

	X = np.asarray(X)

	return X, Y	


if __name__ == '__main__':

	with open('../model_def/bigcldnn.json','r') as f:
		defs = json.load(f)


	dataset_path = defs['dataset_path']
	test_dataset_path = defs['test_dataset_path']
	dims = defs['dims'] 
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

	X, Y = getData(dataset_path)
	test_X, test_Y = getTestData(test_dataset_path)

	print('Read data successfully!')

	model_save_path = defs['model_save_path']
	visual_save_path = defs['visual_save_path']

	model(X, Y, test_X, test_Y)