from keras.layers import Input, Conv1D,MaxPooling1D, Dense,Dropout, Flatten
from keras.layers.merge import Concatenate
from keras.models import Model,load_model,Sequential
from keras import initializers
from keras.constraints import max_norm
from keras.regularizers import l2
from keras.optimizers import Adam
from scipy.io import savemat, loadmat
from keras.callbacks import TensorBoard
import numpy as np
import tables
import csv
from datetime import datetime
import os


ms='/media/taufiq/Data/heart_sound/models/'
fs='/media/taufiq/Data/heart_sound/feature/potes_1DCNN/balancedCV/folds/'
foldname='fold0'

log_name=foldname+ ' ' + str(datetime.now())
log_dir= './logs/' 

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
##########################################
#Creating metadata file for tensorboard
#~ names = [ 'Normal', 'Abnormal' ]
#~ metadata_file=open(os.path.join(log_dir,'metadata.tsv'),'w')
#~ metadata_file.write('Class\n')
#~ for i in range(ytrain.shape[0]):
	#~ metadata_file.write('%s\n' % (names[int(ytrain[i])]))
#~ metadata_file.close()
###########################################

# Model
lr=0.0007
batch_size=8
epoch=50
cnn_thresh=0.4
l2_reg=0.01 # Not specified in paper
random_seed=4
#~ maxnorm=10000.

input1=Input(shape=(2500,1),name='input1')
input2=Input(shape=(2500,1),name='input2')
input3=Input(shape=(2500,1),name='input3')
input4=Input(shape=(2500,1),name='input4')

conv1=Conv1D(8, 5,activation='relu',kernel_initializer=initializers.he_normal(seed=random_seed))(input1)
conv1=MaxPooling1D(pool_size=2)(conv1)
conv1=Conv1D(4, 5,activation='relu',kernel_initializer=initializers.he_normal(seed=random_seed))(conv1)
conv1=MaxPooling1D(pool_size=2)(conv1)
#~ conv1=Dropout(0.25,seed=random_seed)(conv1)
conv1=Flatten()(conv1)

conv2=Conv1D(8, 5,activation='relu',kernel_initializer=initializers.he_normal(seed=random_seed))(input2)
conv2=MaxPooling1D(pool_size=2)(conv2)
conv2=Conv1D(4, 5,activation='relu',kernel_initializer=initializers.he_normal(seed=random_seed))(conv2)
conv2=MaxPooling1D(pool_size=2)(conv2)
#~ conv2=Dropout(0.25,seed=random_seed)(conv2)
conv2=Flatten()(conv2)

conv3=Conv1D(8, 5,activation='relu',kernel_initializer=initializers.he_normal(seed=random_seed))(input3)
conv3=MaxPooling1D(pool_size=2)(conv3)
conv3=Conv1D(4, 5,activation='relu',kernel_initializer=initializers.he_normal(seed=random_seed))(conv3)
conv3=MaxPooling1D(pool_size=2)(conv3)
#~ conv3=Dropout(0.25,seed=random_seed)(conv3)
conv3= Flatten()(conv3)

conv4=Conv1D(8, 5,activation='relu',kernel_initializer=initializers.he_normal(seed=random_seed))(input4)
conv4=MaxPooling1D(pool_size=2)(conv4)
conv4=Conv1D(4, 5,activation='relu',kernel_initializer=initializers.he_normal(seed=random_seed))(conv4)
conv4=MaxPooling1D(pool_size=2)(conv4)
#~ conv4=Dropout(0.25,seed=random_seed)(conv4)
conv4=Flatten()(conv4)
 
merged = Concatenate( axis=1)([conv1,conv2,conv3,conv4])

merged=Dense(20,activation='relu')(merged)
#~ merged=Dropout(0.5)(merged)	
merged=Dense(1,activation='sigmoid')(merged)	

model = Model(inputs=[input1, input2, input3, input4], outputs=merged)

adam = Adam(lr=lr)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

#model=load_model(ms+'potes_train_all')
watchlist=['input1','input2','input3','input4']
log_name=foldname+ ' ' + str(datetime.now())
tensbd=TensorBoard(log_dir='./logs/'+log_name,batch_size=batch_size,write_images=True)


model.fit([x1,x2,x3,x4], ytrain,
 epochs=epoch,
 shuffle=True,
 verbose=1,
 validation_data=([v1,v2,v3,v4],yval),
 batch_size=batch_size,
 callbacks=[tensbd]
 )



model.save(ms+logname)


#y_predict=model.predict([v1,v2,v3,v4],batch_size=batch)
#print(partx.shape)

#tot=partx.shape[1]
#k=0
#y_test=[]
#for i in range(0,tot):
    #temp=np.mean(y_predict[k:(k+int(partx[0,i]))])
    #k=k+int(partx[0,i])
    #if temp>cnn_thresh:
        #y_test.append(1)
    #else:
        #y_test.append(-1)
#y_test=np.array(y_test)      
#sio.savemat(ms+'y_test.mat',{'y_test':y_test,'y_predict':y_predict,'partx':partx})            
    
    
