from keras.layers import Input, Conv1D,MaxPooling1D, Dense,Dropout, Flatten, Activation
from keras import initializers 
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.optimizers import Adam, Nadam, Adamax
from scipy.io import savemat, loadmat
from keras.callbacks import TensorBoard
import numpy as np
import tables
import csv
from datetime import datetime
import os
import argparse
import matplotlib.pyplot as plt

	
def compute_weight(Y,classes):
		num_samples=len(Y)
		n_classes=len(classes)
		num_bin=np.bincount(Y[:,0])
		class_weights={i:(num_samples/(n_classes*num_bin[i])) for i in range(6)}
		return class_weights

def reshape_folds(x_train,x_val,y_train,y_val):
	
	x1=np.transpose(x_train[0,:,:])
	x2=np.transpose(x_train[1,:,:])
	x3=np.transpose(x_train[2,:,:])
	x4=np.transpose(x_train[3,:,:])
	
	x1=np.reshape(x1,[x1.shape[0],x1.shape[1],1])
	x2=np.reshape(x2,[x2.shape[0],x2.shape[1],1])
	x3=np.reshape(x3,[x3.shape[0],x3.shape[1],1])
	x4=np.reshape(x4,[x4.shape[0],x4.shape[1],1])
	
	y_train=np.reshape(y_train,[y_train.shape[0],1])
	
	print x1.shape
	print y_train.shape

	v1=np.transpose(x_val[0,:,:])
	v2=np.transpose(x_val[1,:,:])
	v3=np.transpose(x_val[2,:,:])
	v4=np.transpose(x_val[3,:,:])
	
	v1=np.reshape(v1,[v1.shape[0],v1.shape[1],1])
	v2=np.reshape(v2,[v2.shape[0],v2.shape[1],1])
	v3=np.reshape(v3,[v3.shape[0],v3.shape[1],1])
	v4=np.reshape(v4,[v4.shape[0],v4.shape[1],1])
	
	y_val=np.reshape(y_val,[y_val.shape[0],1])
	
	print v1.shape
	print y_val.shape
	return [x1,x2,x3,x4],y_train,[v1,v2,v3,v4],y_val

########## Parser for arguments (foldname, random_seed, load_path, epochs,
###############################  batch_size)
parser = argparse.ArgumentParser(description='Specify fold to process')
parser.add_argument("fold",
					help="which fold to use from balanced folds generated in /media/taufiq/Data/heart_sound/feature/potes_1DCNN/balancedCV/folds/",
					choices=["fold0","fold1","fold2","fold3"])
parser.add_argument("--seed",type=int,
					help="Random seed for the random number generator (defaults to 1)")
parser.add_argument("--loadmodel",
					help="load previous model checkpoint for retraining (Enter absolute path)")
parser.add_argument("--epochs",type=int,
					help="Number of epochs for training")
parser.add_argument("--batch_size",type=int,
					help="number of minibatches to take during each backwardpass preferably multiple of 2")
parser.add_argument("--verbose",type=int,choices=[1,2],
help="Verbosity mode. 1 = progress bar, 2 = one line per epoch (default 2)")

args = parser.parse_args()
print "{} selected".format(args.fold)
foldname=args.fold

if args.seed:	#	if random seed is specified
	print "Random seed specified as {}".format(args.seed)
	random_seed=args.seed
else:
	random_seed=1

if args.loadmodel: # If a previously trained model is loaded for retraining
	load_path=args.loadmodel #### path to model to be loaded
	
	idx = load_path.find("weights")
	initial_epoch=int(load_path[idx+8:idx+8+4])
	
	print "{} model loaded\nInitial epoch is {}".format(args.loadmodel,initial_epoch)
else:
	print "no model specified, using initializer to initialize weights"
	initial_epoch=0

if args.epochs:	#	if number of training epochs is specified
	print "Training for {} epochs".format(args.epochs)
	epochs=args.epochs
else:
	epochs=200
	print "Training for {} epochs".format(epochs)

if args.batch_size:	#	if batch_size is specified
	print "Training with {} minibatches".format(args.batch_size)
	batch_size=args.batch_size
else:
	batch_size=8
	print "Training with {} minibatches".format(batch_size)

if args.verbose:
	verbose=verbose
else:
	verbose=2
	
#########################################################
if __name__ == '__main__':
	
	
	foldname=foldname
	random_seed=random_seed
	load_path=load_path
	initial_epoch=initial_epoch
	epochs=epochs
	batch_size=batch_size
	verbose=verbose
	
	
	model_dir='/media/taufiq/Data/hear_sound/models/'
	fold_dir='/media/taufiq/Data/heart_sound/feature/potes_1DCNN/balancedCV/folds/'
	log_name=foldname+ ' ' + str(datetime.now())
	log_dir= './logs/' 
	checkpoint_name=model_dir+log_name+'weights.{epoch:04d}-{val_acc:.4f}.hdf5'
	
	eps= 1.1e-5
	bias=False
	l2_reg=0.01
	kernel_size=5
	maxnorm=4.
	dropout_rate=0.25
	padding='valid'
	activation_function='relu'
	subsam=2
	
	lr=0.0007
	lr_decay=1e-8
	lr_reduce_factor=0.5
	patience=4 #for reduceLR
	cooldown=0 #for reduceLR
	
	############## Importing data ############
	
	feat = tables.open_file(fold_dir+foldname+'.mat')
	x_train = feat.root.trainX[:]
	y_train = feat.root.trainY[0,:]
	x_val = feat.root.valX[:]
	y_val = feat.root.valY[0,:]
	
	############## Relabeling ################
	
	for i in range(0,y_train.shape[0]):
		if y_train[i]==-1:
			y_train[i]=0		## Label 0 for normal 1 for abnormal
	for i in range(0,y_val.shape[0]):
		if y_val[i]==-1:
			y_val[i]=0
			
	################### Reshaping ############
	
	x_train,y_train,x_val,y_val=reshape_folds(x_train,x_val,y_train,y_val)
	
	############## Create a model ############
	
	input1=Input(shape=(2500,1),name='input1')
	input2=Input(shape=(2500,1),name='input2')
	input3=Input(shape=(2500,1),name='input3')
	input4=Input(shape=(2500,1),name='input4')
	
	t1 = Conv1D(8, kernel_size=kernel_size,
				kernel_initializer=initializers.he_normal(seed=random_seed),
				padding=padding,
				use_bias=bias,
				kernel_constraint=max_norm(maxnorm),
				kernel_regularizer=l2(l2_reg))(input1)
	t1 = BatchNormalization(epsilon=eps,axis=-1) (t1)
	t1 = Activation(activation_function)(t1)
	t1 = Dropout(rate=dropout_rate,seed=random_seed)(t1)
	t1 = MaxPooling1D(pool_size=subsam)(t1)
	t1 = Conv1D(4, kernel_size=kernel_size,
				kernel_initializer=initializers.he_normal(seed=random_seed),
				padding=padding,
				use_bias=bias,
				kernel_constraint=max_norm(maxnorm),
				kernel_regularizer=l2(l2_reg))(t1)
	t1 = BatchNormalization(epsilon=eps,axis=-1) (t1)
	t1 = Activation(activation_function)(t1)
	t1 = Dropout(rate=dropout_rate,seed=random_seed)(t1)
	t1 = MaxPooling1D(pool_size=subsam)(t1)
	t1 = Flatten()(t1)
	
	t2 = Conv1D(8, kernel_size=kernel_size,
				kernel_initializer=initializers.he_normal(seed=random_seed),
				padding=padding,
				use_bias=bias,
				kernel_constraint=max_norm(maxnorm),
				kernel_regularizer=l2(l2_reg))(input2)
	t2 = BatchNormalization(epsilon=eps,axis=-1) (t2)
	t2 = Activation(activation_function)(t2)
	t2 = Dropout(rate=dropout_rate,seed=random_seed)(t2)
	t2 = MaxPooling1D(pool_size=subsam)(t2)
	t2 = Conv1D(4, kernel_size=kernel_size,
				kernel_initializer=initializers.he_normal(seed=random_seed),
				padding=padding,
				use_bias=bias,
				kernel_constraint=max_norm(maxnorm),
				kernel_regularizer=l2(l2_reg))(t2)
	t2 = BatchNormalization(epsilon=eps,axis=-1) (t2)
	t2 = Activation(activation_function)(t2)
	t2 = Dropout(rate=dropout_rate,seed=random_seed)(t2)
	t2 = MaxPooling1D(pool_size=subsam)(t2)
	t2 = Flatten()(t2)
	
	t3 = Conv1D(8, kernel_size=kernel_size,
				kernel_initializer=initializers.he_normal(seed=random_seed),
				padding=padding,
				use_bias=bias,
				kernel_constraint=max_norm(maxnorm),
				kernel_regularizer=l2(l2_reg))(input3)
	t3 = BatchNormalization(epsilon=eps,axis=-1) (t3)
	t3 = Activation(activation_function)(t3)
	t3 = Dropout(rate=dropout_rate,seed=random_seed)(t3)
	t3 = MaxPooling1D(pool_size=subsam)(t3)
	t3 = Conv1D(4, kernel_size=kernel_size,
				kernel_initializer=initializers.he_normal(seed=random_seed),
				padding=padding,
				use_bias=bias,
				kernel_constraint=max_norm(maxnorm),
				kernel_regularizer=l2(l2_reg))(t3)
	t3 = BatchNormalization(epsilon=eps,axis=-1) (t3)
	t3 = Activation(activation_function)(t3)
	t3 = Dropout(rate=dropout_rate,seed=random_seed)(t3)
	t3 = MaxPooling1D(pool_size=subsam)(t3)
	t3 = Flatten()(t3)	
	
	t4 = Conv1D(8, kernel_size=kernel_size,
				kernel_initializer=initializers.he_normal(seed=random_seed),
				padding=padding,
				use_bias=bias,
				kernel_constraint=max_norm(maxnorm),
				kernel_regularizer=l2(l2_reg))(input4)
	t4 = BatchNormalization(epsilon=eps,axis=-1) (t4)
	t4 = Activation(activation_function)(t4)
	t4 = Dropout(rate=dropout_rate,seed=random_seed)(t4)
	t4 = MaxPooling1D(pool_size=subsam)(t4)
	t4 = Conv1D(4, kernel_size=kernel_size,
				kernel_initializer=initializers.he_normal(seed=random_seed),
				padding=padding,
				use_bias=bias,
				kernel_constraint=max_norm(maxnorm),
				kernel_regularizer=l2(l2_reg))(t4)
	t4 = BatchNormalization(epsilon=eps,axis=-1) (t4)
	t4 = Activation(activation_function)(t4)
	t4 = Dropout(rate=dropout_rate,seed=random_seed)(t4)
	t4 = MaxPooling1D(pool_size=subsam)(t4)
	t4 = Flatten()(t4)		
	print t4
	 
	merged = Concatenate(axis=1)([t1,t2,t3,t4])
	print merged
	
	merged = Dense(20,
		activation=activation_function,
		kernel_initializer=initializers.he_normal(seed=random_seed),
		use_bias=bias,
		kernel_constraint=max_norm(maxnorm),
		kernel_regularizer=l2(l2_reg))(merged)
	merged = BatchNormalization(epsilon=eps,axis=-1)
	merged=Dropout(rate=dropout_rate,seed=random_seed)(merged)	
	merged=Dense(1,activation='sigmoid')(merged)
	
	model = Model(inputs=[input1, input2, input3, input4], outputs=merged)
	adam = Adam(lr=lr,decay=lr_decay)
	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
			

