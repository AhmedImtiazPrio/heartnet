import tensorflow as tf
import numpy as np
from scipy.io import loadmat, savemat
import os
from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = '/home/prio/Keras/datathon/logs'
filename = '/home/prio/Keras/datathon/eog_rk_new_notrans_234rejects_relabeled.mat'
######################## Get the embeddings
Data = loadmat(filename)
X = Data['dat']
Y= Data['hyp']
X = X[0:1000]
Y= Y[0:1000]
Y=Y-1
X = np.reshape(X,(X.shape[0],X.shape[1],1))
X = tf.convert_to_tensor(X)
########################## Create meta data (labels)
names = ['S1', 'S2', 'S3', 'S4', 'REM', 'WAKE' ]
metadata_file=open(os.path.join(LOG_DIR,'metadata.tsv'),'w')
metadata_file.write('Class\n')
for i in range(Y.shape[0]):
	metadata_file.write('%s\n' % (names[int(Y[i])]))
metadata_file.close()
##########################
embedding_var=tf.Variable(X, name='Data')
sess = tf.Session()
sess.run(embedding_var.initializer)
summary_writer=tf.summary.FileWriter(LOG_DIR)
config=projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

embedding.metadata_path= os.path.join(LOG_DIR,'metadata.tsv')

projector.visualize_embeddings(summary_writer, config)
saver = tf.train.Saver([embedding_var])
saver.save(sess, os.path.join(LOG_DIR, 'model2.ckpt'), 1)
sess.close()
