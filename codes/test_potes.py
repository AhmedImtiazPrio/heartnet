import argparse
from potes1DCNN import heartnet, reshape_folds
import tables
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Specify path to test file' )
parser.add_argument("fold", 
	help="file to evaluate from (.mat)")
parser.add_argument("load_path", help="Specify full path to model file (h5py)")

args = parser.parse_args()

foldname=args.fold
load_path=args.load_path
print "importing heart sounds from {}".format(foldname)
print "using model {}".format(load_path)

random_seed=1
load_path=load_path
batch_size=64
print load_path

log_dir= '/media/taufiq/Data/heart_sound/Heart_Sound/codes/logs/'

bn_momentum = 0.99
eps= 1.1e-5
bias=False
l2_reg=0.
l2_reg_dense=0.
kernel_size=5
maxnorm=10000.
dropout_rate=0.5
dropout_rate_dense=0.
padding='valid'
activation_function='relu'
subsam=2

lr=0.0007
lr_decay=1e-8

model = heartnet(activation_function,bn_momentum,bias,dropout_rate,dropout_rate_dense,
		eps,kernel_size,l2_reg,l2_reg_dense,load_path,lr,lr_decay,maxnorm,
		padding,random_seed,subsam)

############# hard coding for now


fold_dir='/media/taufiq/Data/heart_sound/feature/potes_1DCNN/balancedCV/folds/'


feat = tables.open_file(fold_dir+foldname+'.mat')
x_val = feat.root.valX[:]
y_val = feat.root.valY[0,:]
val_parts = feat.root.val_parts[0,:]
y_qual = feat.root.valY[1,:]

for i in range(y_val.shape[0]):
	if y_val[i]==-1:
		y_val[i]=0
		
_,y_qual,x_val,y_val=reshape_folds(np.random.randn(4,4,4),x_val,y_qual,y_val)

y_pred=model.predict(x_val)

true = []
pred=[]
qual=[]
start_idx = 0
print val_parts.shape
for s in val_parts:	
	
	if s==0:		## for e1,32,39,44 in validation0 there was no cardiac cycle
		continue 
	#~ print "part {} start {} stop {}".format(s,start_idx,start_idx+int(s)-1)
					
	temp_ = np.mean(y_val[start_idx:start_idx+int(s)-1])
	temp = np.mean(y_pred[start_idx:start_idx+int(s)-1,:])
	qual_ = np.mean(y_qual[start_idx:start_idx+int(s)-1,:])
	
	pred.append(temp)
	true.append(temp_)
	qual.append(qual_)
	
	start_idx = start_idx + int(s)		

true=np.array(true)
pred=np.array(pred)
qual=np.array(qual)
plt.stem(range(0,true.shape[0]),((true-pred)**2)**.5,'b-')
plt.stem(range(0,qual.shape[0]),qual,'y--',linewidth=.002)
plt.show()
		
		
		
