from keras.layers import Input, Conv1D,MaxPooling1D, Dense,Dropout, Flatten
from keras.layers.merge import Concatenate
from keras.models import Model,load_model,Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from scipy.io import savemat, loadmat
from keras.callbacks import TensorBoard
import numpy as np
import tables
import csv
from datetime import datetime
import os
import argparse

########## Parser for arguments (foldname, random_seed, load_path, epochs,
###############################  batch_size)
parser = argparse.ArgumentParser(description='Specify fold to process')
parser.add_argument("fold",
					help="which fold to use from balanced folds generated in /media/taufiq/Data/heart_sound/feature/potes_1DCNN/balancedCV/folds/",
					choices=["fold0","fold1","fold2","fold3"])
parser.add_argument("--seed",type=int,
					help="Random seed for the random number generator (defaults to zero)")
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
	random_seed=0

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
else
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
	fs='/media/taufiq/Data/heart_sound/feature/potes_1DCNN/balancedCV/folds/'
	log_name=foldname+ ' ' + str(datetime.now())
	log_dir= './logs/' 
	checkpoint_name=model_dir+log_name+'weights.{epoch:04d}-{val_acc:.4f}.hdf5'
	
	bias=False
	l2_reg=0.01
	kernel_size=16
	maxnorm=4.
	dropout_rate=0.25
	
	lr=0.0007
	lr_decay=1e-8
	lr_reduce_factor=0.5 
	patience=4 #for reduceLR
	cooldown=0 #for reduceLR
	
	feat = tables.open_file(fs+foldname+'.mat')
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
	##########################################
	print(x_train.shape)
	
	bands=x_train.shape[0]
	cols=x_train.shape[1]
	files=x_train.shape[2]
	
	# Reshape to channels_last
	ytrain=np.zeros((files,1))
	x1=np.zeros((files,cols,1))
	x2=np.zeros((files,cols,1))
	x3=np.zeros((files,cols,1))
	x4=np.zeros((files,cols,1))
	
	for i in range (0,files):
	    ytrain[i,0]=y_train[i]
	    
	for i in range (0,files):
	    for j in range (0,cols):
	        x1[i,j,0]=x_train[0,j,i]
	        x2[i,j,0]=x_train[1,j,i]
	        x3[i,j,0]=x_train[2,j,i]
	        x4[i,j,0]=x_train[3,j,i]       
	print(x1.shape)
	#########################################
	print(x_val.shape)
	
	bands=x_val.shape[0]
	cols=x_val.shape[1]
	files=x_val.shape[2]
	
	# Reshape to channels_last
	yval=np.zeros((files,1))
	v1=np.zeros((files,cols,1))
	v2=np.zeros((files,cols,1))
	v3=np.zeros((files,cols,1))
	v4=np.zeros((files,cols,1))
	
	for i in range (0,files):
	    yval[i,0]=y_val[i]
	    
	for i in range (0,files):
	    for j in range (0,cols):
	        v1[i,j,0]=x_val[0,j,i]
	        v2[i,j,0]=x_val[1,j,i]
	        v3[i,j,0]=x_val[2,j,i]
	        v4[i,j,0]=x_val[3,j,i]       
	print(v1.shape)
	
	
	
	

def compute_weight(Y,classes):
		num_samples=len(Y)
		n_classes=len(classes)
		num_bin=np.bincount(Y[:,0])
		class_weights={i:(num_samples/(n_classes*num_bin[i])) for i in range(6)}
		return class_weights

