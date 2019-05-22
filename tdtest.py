import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import math


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
		x_fc = tf.reshape(pool4,shape = [-1, 324])
	
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
		x_fc = tf.reshape(pool4,shape = [-1, 324])
	
	return x_fc


def tie_input_timesteps(xs):
	return tf.stack([xi for xi in xs])


def lstm_layers(x_fcs_input):
	with tf.variable_scope('lstm'):
		lstm_layer = tf.contrib.rnn.LSTMCell(64)
		outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_layer, lstm_layer, x_fcs_input, dtype=tf.float32)

	return outputs



def fc_layers(x_fc):
	with tf.variable_scope('fc'):
		w1 = tf.get_variable("w1",[fc_size1,fc_size2],initializer = tf.contrib.layers.xavier_initializer())
		b1 = tf.get_variable("b1", [fc_size2], initializer = tf.zeros_initializer())
		w2 = tf.get_variable("w2",[fc_size2,fc_size3],initializer = tf.contrib.layers.xavier_initializer())
		b2 = tf.get_variable("b2", [fc_size3], initializer = tf.zeros_initializer())
	
		tf.summary.histogram('w1', w1)
		tf.summary.histogram('b1', b1)
		tf.summary.histogram('w2', w2)
		tf.summary.histogram('b2', b2)

		z1 = tf.add(tf.matmul(x_fc,w1),b1)
		a1 = tf.nn.relu(z1)
		z2 = tf.add(tf.matmul(a1,w2),b2)

	return z2


'''def fc_layers(x_fc):
	w1 = tf.get_variable("w1",[fc_size1,fc_size2],initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1", [fc_size2], initializer = tf.zeros_initializer())
	w2 = tf.get_variable("w2",[fc_size2,fc_size3],initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", [fc_size3], initializer = tf.zeros_initializer())
	w3 = tf.get_variable("w3",[fc_size3,fc_size4],initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable("b3", [fc_size4], initializer = tf.zeros_initializer())
	w4 = tf.get_variable("w4",[fc_size4,fc_size5],initializer = tf.contrib.layers.xavier_initializer())
	b4 = tf.get_variable("b4", [fc_size5], initializer = tf.zeros_initializer())
	w5 = tf.get_variable("w5",[fc_size5,fc_size6],initializer = tf.contrib.layers.xavier_initializer())
	b5 = tf.get_variable("b5", [fc_size6], initializer = tf.zeros_initializer())
	w6 = tf.get_variable("w6",[fc_size6,fc_size7],initializer = tf.contrib.layers.xavier_initializer())
	b6 = tf.get_variable("b6", [fc_size7], initializer = tf.zeros_initializer())
	w7 = tf.get_variable("w7",[fc_size7,fc_size8],initializer = tf.contrib.layers.xavier_initializer())
	b7 = tf.get_variable("b7", [fc_size8], initializer = tf.zeros_initializer())

	with tf.name_scope('fc'):	
		z1 = tf.add(tf.matmul(x_fc,w1),b1)
		a1 = tf.nn.relu(z1)
		z2 = tf.add(tf.matmul(a1,w2),b2)
		a2 = tf.nn.relu(z2)
		z3 = tf.add(tf.matmul(a2,w3),b3)
		a3 = tf.nn.relu(z3)
		z4 = tf.add(tf.matmul(a3,w4),b4)
		a4 = tf.nn.relu(z4)
		z5 = tf.add(tf.matmul(a4,w5),b5)
		a5 = tf.nn.relu(z5)
		z6 = tf.add(tf.matmul(a5,w6),b6)
		a6 = tf.nn.relu(z6)
		z7 = tf.add(tf.matmul(a6,w7),b7)

	return z7'''


	
def model(X, Y):
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
		x_fcs_output_reshaped = tf.reshape(x_fcs_output,[-1,5*128])

	z2 = fc_layers(x_fcs_output_reshaped)
	
	with tf.variable_scope('cost'):
		cost = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z2,labels=y,name='loss'))

	with tf.variable_scope('accuracy'):	
		equality = tf.equal(tf.argmax(z2, 1), tf.argmax(y, 1))
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
	#saver.restore(sess,'D:/fyndML/Models/cldnn.ckpt')

	##Tensorboard data
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter('visual/tdtest')
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
			e_sum, _, e_cost = sess.run([merged_summary,optimiser,cost],feed_dict = {
				x[0]:batch_X_1,
				x[1]:batch_X_2,
				x[2]:batch_X_2,
				x[3]:batch_X_4,
				x[4]:batch_X_5,
				y:batch_Y
				})
			writer.add_summary(e_sum, i)
			#print(e_z2)
			#print(Y)
			print('Cost after epoch ' + str(epoch + 1) + ', batch ' + str(batch + 1) + ' = ' + str(e_cost))

			if batch % 10 == 0:
				print('Saving checkpoint...')
				#saver.save(sess,'D:/fyndML/Models/tdtest.ckpt')
				print('Done saving!')

		#saver.save(sess,'D:/fyndML/Models/tdtest.ckpt')
	


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
			image = np.array(image).reshape(200,200,1)
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



if __name__ == '__main__':
	dataset_path = os.getcwd() + '/Dataset/fyndDataGray.csv'
	dims = 200
	output_classes = 6
	epochs = 20
	batch_size = 128
	dataset_size = 250
	batches = math.ceil(dataset_size / batch_size)
	
	ground_truth = {
		'zipper' : 0,
		'backstrap' : 1,
		'slip_on' : 2,
		'lace_up' : 3,
		'buckle' : 4,
		'hook&look' : 5,
	}

	conv1_fmaps = 32
	conv1_ksize = 4
	conv1_stride = 1
	conv1_pad = 'valid'

	conv2_fmaps = 16
	conv2_ksize = 4
	conv2_stride = 1
	conv2_pad = 'valid'

	conv3_fmaps = 8
	conv3_ksize = 4
	conv3_stride = 1
	conv3_pad = 'valid'

	conv4_fmaps = 4
	conv4_ksize = 4
	conv4_stride = 1
	conv4_pad = 'valid'

	fc_dims = 640

	fc_size1 = fc_dims
	fc_size2 = 100
	fc_size3 = output_classes

	'''
	fc_size1 = fc_dims
	fc_size2 = 150
	fc_size3 = 70
	fc_size4 = 30
	fc_size5 = output_classes
	'''

	X, Y = getData(dataset_path)

	print('Read data successfully!')

	model(X, Y)