import argparse
from potes1DCNN import heartnet

parser = argparse.ArgumentParser(description='Specify path to test file' )
parser.add_argument("evaluate_mat", 
	help="file to evaluate from (.mat)")
parser.add_argument("load_path", help="Specify full path to model file (h5py)")

args = parser.parse_args()

evaluate_mat=args.evaluate_mat
load_path=args.load_path
print "importing heart sounds from {}".format(folder)
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

