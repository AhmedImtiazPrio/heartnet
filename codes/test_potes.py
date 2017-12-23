import argparse
from potes1DCNN import heartnet

parser = argparse.ArgumentParser(description='Specify path to test files' )
parser.add_argument("folder", 
	help="folder to read all the heart sound files from and evaluate")
parser.add_argument("model_path", help="Specify full path to model file (h5py)")

args = parser.parse_args()

folder=args.folder
model_path=args.model_path
print "importing heart sounds from {}".format(folder)
print "using model {}".format(model_path)

random_seed=1
load_path=load_path
batch_size=batch_size

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

model = hearnet(activation_function,bn_momentum,bias,dropout_rate,dropout_rate_dense,
		eps,kernel_size,l2_reg,l2_reg_dense,load_path,lr,lr_decay,maxnorm,
		padding,random_seed,subsam)

os.system()
