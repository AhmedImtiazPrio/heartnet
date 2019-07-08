from __future__ import print_function, division, absolute_import
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))
# from clr_callback import CyclicLR
# import dill
from AudioDataGenerator import BalancedAudioDataGenerator
import os
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import pandas as pd
import tables
from datetime import datetime
import argparse
from keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.utils import plot_model
from heartnet_v1 import log_macc, write_meta, compute_weight, reshape_folds, results_log, lr_schedule
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from modules import heartnetTop
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.optimizers import Adam as optimizer
from keras.layers import Dense,Flatten,Dropout
from keras.initializers import he_normal as initializer
from utils import load_data, sessionLog

if __name__ == '__main__':

    ### Parser for arguments (HS, random_seed, load_path, epochs, batch_size)
    parser = argparse.ArgumentParser()
    parser.add_argument("HS",
                        help="input HS tensor")
    parser.add_argument("--seed", type=int,
                        help="Random seed for the random number generator (defaults to 1)")
    parser.add_argument("--loadmodel",
                        help="load previous model checkpoint for retraining (Enter absolute path)")
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int,
                        help="number of minibatches to take during each backwardpass preferably multiple of 2")
    parser.add_argument("--verbose", type=int, choices=[1, 2],
                        help="Verbosity mode. 1 = progress bar, 2 = one line per epoch (default 2)")
    parser.add_argument("--comment",
                        help = "Add comments to the log files")
    parser.add_argument("--type")
    parser.add_argument("--lr", type=float)

    args = parser.parse_args()
    print("%s selected" % (args.HS))
    HS = args.HS

    if args.seed:  # if random seed is specified
        print("Random seed specified as %d" % (args.seed))
        random_seed = args.seed
    else:
        random_seed = 1

    if args.loadmodel:  # If a previously trained model is loaded for retraining
        load_path = args.loadmodel  #### path to model to be loaded

        idx = load_path.find("weights")
        initial_epoch = int(load_path[idx + 8:idx + 8 + 4])

        print("%s model loaded\nInitial epoch is %d" % (args.loadmodel, initial_epoch))
    else:
        print("no model specified, using initializer to initialize weights")
        initial_epoch = 0
        load_path = False

    if args.epochs:  # if number of training epochs is specified
        print("Training for %d epochs" % (args.epochs))
        epochs = args.epochs
    else:
        epochs = 200
        print("Training for %d epochs" % (epochs))

    if args.batch_size:  # if batch_size is specified
        print("Training with %d samples per minibatch" % (args.batch_size))
        batch_size = args.batch_size
    else:
        batch_size = 64
        print("Training with %d minibatches" % (batch_size))

    if args.verbose:
        verbose = args.verbose
        print("Verbosity level %d" % (verbose))
    else:
        verbose = 1
    if args.comment:
        comment = args.comment
    else:
        comment = None
    if args.type:
        FIR_type=args.type
    else:
        FIR_type=1
    if args.lr:
        lr= args.lr
    else:
        lr=0.0012843784


    #########################################################

    model_dir = os.path.join('..','models')
    data_dir = os.path.join('..','data')
    log_name = HS + ' ' + str(datetime.now()).replace(':','-')
    log_dir = os.path.join('..','logs')
    if not os.path.exists(os.path.join(model_dir,log_name)):
        os.makedirs(os.path.join(model_dir,log_name))
    checkpoint_name = os.path.join(model_dir,log_name,'weights.{epoch:04d}-{val_acc:.4f}.hdf5')
    results_path = os.path.join('..','logs','resultsLog.csv')

    ### Init Params

    params = {

        'num_filt': (8, 4),
        'num_dense': 20,
        'bn_momentum': 0.99,
        'eps': 1.1e-5,
        'bias': False,
        'l2_reg': 0.04864911065093751,
        'l2_reg_dense': 0.,
        'kernel_size': 5,
        'maxnorm': 10000.,
        'dropout_rate': 0.5,
        'dropout_rate_dense': 0.,
        'padding': 'valid',
        'activation_function': 'relu',
        'subsam': 2,
        'FIR_train': True,
        'trainable': True,
        'decision': 'majority',
        'lr':lr,
        'lr_decay': 0.0001132885*(batch_size/64),
        'random_seed':random_seed,
        'initializer':initializer,
        'optimizer':optimizer,

    }

    x_train,y_train,train_files,train_parts,x_val,y_val,val_files,val_parts = load_data(HS,data_dir)
    train_subset = np.asarray([each[0] for each in train_files])
    val_subset = np.asarray([each[0] for each in val_files])

    topModel = heartnetTop(**params)
    out = Flatten()(topModel.output)
    out = Dense(20,activation=params['activation_function'],
                kernel_initializer=initializer(seed=random_seed),
                use_bias=True,kernel_regularizer=l2(params['l2_reg_dense']))(out)
    out = Dropout(rate=params['dropout_rate_dense'], seed=random_seed)(out)
    out = Dense(2, activation='softmax')(out)
    model = Model(inputs=topModel.input, outputs=out)

    if load_path:
        model.load_weights(filepath=load_path, by_name=False)

    adam = optimizer(lr=params['lr'], decay=params['lr_decay'])
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    if verbose:
        model.summary()

    # plot_model(model, to_file=os.path.join(model_dir,log_name,'model.png'),show_shapes=True)
    model_json = model.to_json()

    with open(os.path.join(model_dir,log_name,'model.json'), "w") as json_file:
        json_file.write(model_json)

    modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
                                    monitor='val_acc', save_best_only=False, mode='max')
    tensbd = TensorBoard(log_dir=os.path.join(log_dir,log_name),
                         batch_size=batch_size,
                         write_images=False)
    csv_logger = CSVLogger(os.path.join(log_dir,log_name,'training.csv'))

    ######### Data Generator ############

    datagen = BalancedAudioDataGenerator(shift=.1)

    meta_labels = np.asarray([ord(each) - 97 for each in train_subset])
    for idx, each in enumerate(np.unique(train_subset)):
        meta_labels[np.where(np.logical_and(y_train[:, 0] == 1, np.asarray(train_subset) == each))] = 6 + idx

    flow = datagen.flow(x_train, y_train,
                        meta_label=meta_labels,
                        batch_size=batch_size, shuffle=True,
                        seed=random_seed)
    try:
        model.fit_generator(flow,
                            steps_per_epoch= sum(np.asarray(train_subset) == 'a') // flow.chunk_size,
                            use_multiprocessing=False,
                            epochs=epochs,
                            verbose=verbose,
                            shuffle=True,
                            callbacks=[modelcheckpnt,
                                       log_macc(val_parts, decision=params['decision'],verbose=verbose, val_files=val_subset),
                                       tensbd, csv_logger],
                            validation_data=(x_val, y_val),
                            initial_epoch=initial_epoch,
                            )

        sessionLog(results_path=results_path, log_dir=log_dir, log_name=log_name, batch_size=batch_size, verbose=verbose,
                   comment=comment, **params)

    except KeyboardInterrupt:
        sessionLog(results_path=results_path, log_dir=log_dir, log_name=log_name, batch_size=batch_size, verbose=verbose,
                   comment=comment, **params)