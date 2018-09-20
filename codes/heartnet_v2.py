from __future__ import print_function, division, absolute_import
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))
# from clr_callback import CyclicLR
# import dill
from AudioDataGenerator import AudioDataGenerator
import os
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import pandas as pd
import tables
from datetime import datetime
import argparse
from keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Activation, AveragePooling1D
from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.optimizers import Adam  # Nadam, Adamax
from keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.utils import plot_model
from custom_layers import Conv1D_zerophase_linear, Conv1D_linearphase, Conv1D_zerophase,\
    DCT1D, Conv1D_gammatone, Conv1D_linearphaseType
from heartnet_v1 import log_macc, write_meta, compute_weight, reshape_folds, results_log
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def branch(input_tensor,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable):

    num_filt1, num_filt2 = num_filt
    t = Conv1D(num_filt1, kernel_size=kernel_size,
                kernel_initializer=initializers.he_normal(seed=random_seed),
                padding=padding,
                use_bias=bias,
                kernel_constraint=max_norm(maxnorm),
                trainable=trainable,
                kernel_regularizer=l2(l2_reg))(input_tensor)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    t = Activation(activation_function)(t)
    t = Dropout(rate=dropout_rate, seed=random_seed)(t)
    t = MaxPooling1D(pool_size=subsam)(t)
    t = Conv1D(num_filt2, kernel_size=kernel_size,
               kernel_initializer=initializers.he_normal(seed=random_seed),
               padding=padding,
               use_bias=bias,
               trainable=trainable,
               kernel_constraint=max_norm(maxnorm),
               kernel_regularizer=l2(l2_reg))(t)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    t = Activation(activation_function)(t)
    t = Dropout(rate=dropout_rate, seed=random_seed)(t)
    t = MaxPooling1D(pool_size=subsam)(t)
    # t = Flatten()(t)
    return t

def heartnet(load_path,activation_function='relu', bn_momentum=0.99, bias=False, dropout_rate=0.5, dropout_rate_dense=0.0,
             eps=1.1e-5, kernel_size=5, l2_reg=0.0, l2_reg_dense=0.0,lr=0.0012843784, lr_decay=0.0001132885, maxnorm=10000.,
             padding='valid', random_seed=1, subsam=2, num_filt=(8, 4), num_dense=20,FIR_train=False,trainable=True):

    input = Input(shape=(2500, 1))

    coeff_path = '/media/taufiq/Data1/heart_sound/heartnetTransfer/filterbankcoeff60.mat'
    coeff = tables.open_file(coeff_path)
    b1 = coeff.root.b1[:]
    b1 = np.hstack(b1)
    b1 = np.reshape(b1, [b1.shape[0], 1, 1])

    b2 = coeff.root.b2[:]
    b2 = np.hstack(b2)
    b2 = np.reshape(b2, [b2.shape[0], 1, 1])

    b3 = coeff.root.b3[:]
    b3 = np.hstack(b3)
    b3 = np.reshape(b3, [b3.shape[0], 1, 1])

    b4 = coeff.root.b4[:]
    b4 = np.hstack(b4)
    b4 = np.reshape(b4, [b4.shape[0], 1, 1])

    ## Conv1D_linearphase

    # input1 = Conv1D_linearphase(1 ,61, use_bias=False,
    #                 # kernel_initializer=initializers.he_normal(random_seed),
    #                 weights=[b1[30:]],
    #                 padding='same',trainable=FIR_train)(input)
    # input2 = Conv1D_linearphase(1, 61, use_bias=False,
    #                 # kernel_initializer=initializers.he_normal(random_seed),
    #                 weights=[b2[30:]],
    #                 padding='same',trainable=FIR_train)(input)
    # input3 = Conv1D_linearphase(1, 61, use_bias=False,
    #                 # kernel_initializer=initializers.he_normal(random_seed),
    #                 weights=[b3[30:]],
    #                 padding='same',trainable=FIR_train)(input)
    # input4 = Conv1D_linearphase(1, 61, use_bias=False,
    #                 # kernel_initializer=initializers.he_normal(random_seed),
    #                 weights=[b4[30:]],
    #                 padding='same',trainable=FIR_train)(input)

    # ## Conv1D_linearphase Anti-Symmetric
    #
    # input1 = Conv1D_linearphaseType(1 ,61, use_bias=False,
    #                 # kernel_initializer=initializers.he_normal(random_seed),
    #                 weights=[b1[30:]],
    #                 padding='same',trainable=FIR_train, type = 3)(input)
    # input2 = Conv1D_linearphaseType(1, 61, use_bias=False,
    #                 # kernel_initializer=initializers.he_normal(random_seed),
    #                 weights=[b2[30:]],
    #                 padding='same',trainable=FIR_train, type = 3)(input)
    # input3 = Conv1D_linearphaseType(1, 61, use_bias=False,
    #                 # kernel_initializer=initializers.he_normal(random_seed),
    #                 weights=[b3[30:]],
    #                 padding='same',trainable=FIR_train, type = 3)(input)
    # input4 = Conv1D_linearphaseType(1, 61, use_bias=False,
    #                 # kernel_initializer=initializers.he_normal(random_seed),
    #                 weights=[b4[30:]],
    #                 padding='same',trainable=FIR_train, type = 3)(input)

    #Conv1D_gammatone

    input1 = Conv1D_gammatone(kernel_size=81,filters=1,fsHz=1000,use_bias=False,padding='same')(input)
    input2 = Conv1D_gammatone(kernel_size=81,filters=1,fsHz=1000,use_bias=False,padding='same')(input)
    input3 = Conv1D_gammatone(kernel_size=81,filters=1,fsHz=1000,use_bias=False,padding='same')(input)
    input4 = Conv1D_gammatone(kernel_size=81,filters=1,fsHz=1000,use_bias=False,padding='same')(input)

    t1 = branch(input1,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t2 = branch(input2,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t3 = branch(input3,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t4 = branch(input4,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)

    merged = Concatenate(axis=-1)([t1, t2, t3, t4])
    # merged = DCT1D()(merged)
    merged = Flatten()(merged)
    merged = Dense(num_dense,
                   activation=activation_function,
                   kernel_initializer=initializers.he_normal(seed=random_seed),
                   use_bias=bias,
                   kernel_constraint=max_norm(maxnorm),
                   kernel_regularizer=l2(l2_reg_dense))(merged)
    # merged = BatchNormalization(epsilon=eps,momentum=bn_momentum,axis=-1) (merged)
    # merged = Activation(activation_function)(merged)
    merged = Dropout(rate=dropout_rate_dense, seed=random_seed)(merged)
    merged = Dense(2, activation='softmax')(merged)

    model = Model(inputs=input, outputs=merged)

    if load_path:  # If path for loading model was specified
        model.load_weights(filepath=load_path, by_name=False)

    adam = Adam(lr=lr, decay=lr_decay)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    try:
        ########## Parser for arguments (foldname, random_seed, load_path, epochs, batch_size)
        parser = argparse.ArgumentParser(description='Specify fold to process')
        parser.add_argument("fold",
                            help="which fold to use from balanced folds generated in /media/taufiq/Data/heart_sound/feature/potes_1DCNN/balancedCV/folds/")
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
        parser.add_argument("--classweights", type=bool,
                            help="if True, class weights are added according to the ratio of the two classes present in the training data")
        parser.add_argument("--comment",
                            help = "Add comments to the log files")

        args = parser.parse_args()
        print("%s selected" % (args.fold))
        foldname = args.fold

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
            verbose = 2
        if args.classweights:
            addweights = True
        else:
            addweights = False
        if args.comment:
            comment = args.comment
        else:
            comment = None


        #########################################################

        foldname = foldname
        random_seed = random_seed
        load_path = load_path
        initial_epoch = initial_epoch
        epochs = epochs
        batch_size = batch_size
        verbose = verbose

        model_dir = '/media/taufiq/Data1/heart_sound/models/'
        fold_dir = '/media/taufiq/Data1/heart_sound/feature/segmented_noFIR/'
        log_name = foldname + ' ' + str(datetime.now())
        log_dir = '/media/taufiq/Data1/heart_sound/logs/'
        if not os.path.exists(model_dir + log_name):
            os.makedirs(model_dir + log_name)
        checkpoint_name = model_dir + log_name + "/" + 'weights.{epoch:04d}-{val_acc:.4f}.hdf5'
        results_path = '/media/taufiq/Data1/heart_sound/results_2class.csv'

        num_filt = (8, 4)
        num_dense = 20

        bn_momentum = 0.99
        eps = 1.1e-5
        bias = False
        l2_reg = 0.04864911065093751
        l2_reg_dense = 0.
        kernel_size = 5
        maxnorm = 10000.
        dropout_rate = 0.5
        dropout_rate_dense = 0.
        padding = 'valid'
        activation_function = 'relu'
        subsam = 2
        FIR_train= True
        trainable = True
        decision = 'majority'  # Decision algorithm for inference over total recording ('majority','confidence')

        lr =  0.0012843784 ## After bayesian optimization

        ###### lr_decay optimization ######
        lr_decay =0.0001132885
        # lr_decay =3.64370733503E-06
        # lr_decay =3.97171548784E-08
        ###################################



        lr_reduce_factor = 0.5
        patience = 4  # for reduceLR
        cooldown = 0  # for reduceLR
        res_thresh = 0.5  # threshold for turning probability values into decisions

        ############## Importing data ############

        feat = tables.open_file(fold_dir + foldname + '.mat')
        x_train = feat.root.trainX[:]
        y_train = feat.root.trainY[0, :]
        x_val = feat.root.valX[:]
        y_val = feat.root.valY[0, :]
        train_parts = feat.root.train_parts[:]
        val_parts = feat.root.val_parts[0, :]

        ############## Relabeling ################

        for i in range(0, y_train.shape[0]):
            if y_train[i] == -1:
                y_train[i] = 0  ## Label 0 for normal 1 for abnormal
        for i in range(0, y_val.shape[0]):
            if y_val[i] == -1:
                y_val[i] = 0

        ############# Parse Database names ########

        train_files = []
        for each in feat.root.train_files[:][0]:
            train_files.append(chr(each))
        print(len(train_files))
        val_files = []
        for each in feat.root.val_files[:][0]:
            val_files.append(chr(each))
        print(len(val_files))

        ################### Reshaping ############

        x_train, y_train, x_val, y_val = reshape_folds(x_train, x_val, y_train, y_val)
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)
        ############### Write metadata for embedding visualizer ############

        # metadata_file = write_meta(y_val,log_dir)

        ############## Create a model ############

        model = heartnet(load_path,activation_function, bn_momentum, bias, dropout_rate, dropout_rate_dense,
                         eps, kernel_size, l2_reg, l2_reg_dense, lr, lr_decay, maxnorm,
                         padding, random_seed, subsam, num_filt, num_dense, FIR_train, trainable)
        model.summary()
        plot_model(model, to_file='model.png', show_shapes=True)
        model_json = model.to_json()
        with open(model_dir + log_name+"/model.json", "w") as json_file:
            json_file.write(model_json)
        # embedding_layer_names =set(layer.name
        #                     for layer in model.layers
        #                     if (layer.name.startswith('dense_')))
        # print(embedding_layer_names)

        ####### Define Callbacks ######

        modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
                                        monitor='val_acc', save_best_only=False, mode='max')
        tensbd = TensorBoard(log_dir=log_dir + log_name,
                             batch_size=batch_size, histogram_freq = 3,
                             write_grads=True,
                             # embeddings_freq=99,
                             # embeddings_layer_names=embedding_layer_names,
                             # embeddings_data=x_val,
                             # embeddings_metadata=metadata_file,
                             write_images=False)
        csv_logger = CSVLogger(log_dir + log_name + '/training.csv')

        ######### Data Generator ############

        datagen = AudioDataGenerator(
                                     shift=.1,
                                     # roll_range=.1,
                                     # fill_mode='reflect',
                                     # featurewise_center=True,
                                     # zoom_range=.1,
                                     # zca_whitening=True,
                                     # samplewise_center=True,
                                     # samplewise_std_normalization=True,
                                     )
        # valgen = AudioDataGenerator(
        #     # fill_mode='reflect',
        #     # featurewise_center=True,
        #     # zoom_range=.2,
        #     # zca_whitening=True,
        #     # samplewise_center=True,
        #     # samplewise_std_normalization=True,
        # )

        model.fit_generator(datagen.flow(x_train, y_train, batch_size, shuffle=True, seed=random_seed),
                            steps_per_epoch=len(x_train) // batch_size,
                            # max_queue_size=20,
                            use_multiprocessing=False,
                            epochs=epochs,
                            verbose=verbose,
                            shuffle=True,
                            callbacks=[modelcheckpnt,
                                       log_macc(val_parts, decision=decision,verbose=verbose, val_files=val_files),
                                       tensbd, csv_logger],
                            validation_data=(x_val, y_val),
                            )

        ######### Run forest run!! ##########

        # if addweights:  ## if input arg classweights was specified True
        #
        #     class_weight = compute_weight(y_train, np.unique(y_train))
        #
        #     model.fit(x_train, y_train,
        #               batch_size=batch_size,
        #               epochs=epochs,
        #               shuffle=True,
        #               verbose=verbose,
        #               validation_data=(x_val, y_val),
        #               callbacks=[modelcheckpnt,
        #                          log_macc(val_parts, decision=decision,verbose=verbose, val_files=val_files),
        #                          tensbd, csv_logger],
        #               initial_epoch=initial_epoch,
        #               class_weight=class_weight)
        #
        # else:
        #
        #     model.fit(x_train, y_train,
        #               batch_size=batch_size,
        #               epochs=epochs,
        #               shuffle=True,
        #               verbose=verbose,
        #               validation_data=(x_val, y_val),
        #               callbacks=[
        #                         # CyclicLR(base_lr=0.0001132885,
        #                         #          max_lr=0.0012843784,
        #                         #          step_size=8*(x_train.shape[0]//batch_size),
        #                         #          ),
        #                         modelcheckpnt,
        #                         log_macc(val_parts, decision=decision,verbose=verbose, val_files=val_files),
        #                         tensbd, csv_logger],
        #               initial_epoch=initial_epoch)

        ############### log results in csv ###############
        plot_model(model, to_file=log_dir + log_name + '/model.png', show_shapes=True)
        results_log(results_path=results_path, log_dir=log_dir, log_name=log_name, activation_function=activation_function, addweights=addweights, kernel_size=kernel_size, maxnorm=maxnorm,
                    dropout_rate=dropout_rate, dropout_rate_dense=dropout_rate_dense, l2_reg=l2_reg,
                    l2_reg_dense=l2_reg_dense, batch_size=batch_size, lr=lr, bn_momentum=bn_momentum, lr_decay=lr_decay,
                    num_dense=num_dense, comment=comment,num_filt=num_filt)
        print(model.layers[1].get_weights())
        with K.get_session() as sess:
            impulse_gammatone = sess.run(model.layers[1].impulse_gammatone())
        plt.plot(impulse_gammatone)
        plt.show()


    except KeyboardInterrupt:
        ############ If ended in advance ###########
        plot_model(model, to_file=log_dir + log_name + '/model.png', show_shapes=True)
        results_log(results_path=results_path, log_dir=log_dir, log_name=log_name, activation_function=activation_function, addweights=addweights, kernel_size=kernel_size, maxnorm=maxnorm,
                    dropout_rate=dropout_rate, dropout_rate_dense=dropout_rate_dense, l2_reg=l2_reg,
                    l2_reg_dense=l2_reg_dense, batch_size=batch_size, lr=lr, bn_momentum=bn_momentum, lr_decay=lr_decay,
                    num_dense=num_dense, comment=comment,num_filt=num_filt)
        model_json = model.to_json()
        with open(model_dir + log_name+"/model.json", "w") as json_file:
            json_file.write(model_json)
