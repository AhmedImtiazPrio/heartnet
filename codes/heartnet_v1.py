from __future__ import print_function, division, absolute_import
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))
# from clr_callback import CyclicLR
import dill
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
from keras.optimizers import Adam#, Nadam, Adamax
from keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.utils import plot_model
from custom_layers import Conv1D_zerophase_linear, Conv1D_linearphase, Conv1D_zerophase
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
# import matplotlib.pyplot as plt

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
    # t = Conv1D(num_filt2, kernel_size=kernel_size,
    #            kernel_initializer=initializers.he_normal(seed=random_seed),
    #            padding=padding,
    #            use_bias=bias,
    #            trainable=trainable,
    #            kernel_constraint=max_norm(maxnorm),
    #            kernel_regularizer=l2(l2_reg))(t)
    # t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    # t = Activation(activation_function)(t)
    # t = Dropout(rate=dropout_rate, seed=random_seed)(t)
    # t = MaxPooling1D(pool_size=subsam)(t)
    t = Flatten()(t)
    return t

def write_meta(Y,log_dir):
    names = ['Normal', 'Abnormal', 'Normal']
    metadata_file = open(os.path.join(log_dir, 'metadata.tsv'), 'w')
    # metadata_file.write('Class\n') ## For single metadata, headers are not required
    for i in range(Y.shape[0]):
        metadata_file.write('%s\n' % (names[int(Y[i])]))
    metadata_file.close()
    return os.path.join(log_dir, 'metadata.tsv')

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

    input1 = Conv1D_linearphase(1 ,61, use_bias=False,
                    # kernel_initializer=initializers.he_normal(random_seed),
                    weights=[b1[30:]],
                    padding='same',trainable=FIR_train)(input)
    input2 = Conv1D_linearphase(1, 61, use_bias=False,
                    # kernel_initializer=initializers.he_normal(random_seed),
                    weights=[b2[30:]],
                    padding='same',trainable=FIR_train)(input)
    input3 = Conv1D_linearphase(1, 61, use_bias=False,
                    # kernel_initializer=initializers.he_normal(random_seed),
                    weights=[b3[30:]],
                    padding='same',trainable=FIR_train)(input)
    input4 = Conv1D_linearphase(1, 61, use_bias=False,
                    # kernel_initializer=initializers.he_normal(random_seed),
                    weights=[b4[30:]],
                    padding='same',trainable=FIR_train)(input)

    t1 = branch(input1,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t2 = branch(input2,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t3 = branch(input3,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t4 = branch(input4,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)

    merged = Concatenate(axis=1)([t1, t2, t3, t4])
    # merged = Dropout(rate=dropout_rate_dense, seed=random_seed)(merged)
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


class log_macc(Callback):

    def __init__(self, val_parts,decision='majority',verbose=0, val_files=None):
        super(log_macc, self).__init__()
        self.val_parts = val_parts
        self.decision = decision
        self.verbose = verbose
        self.val_files = np.asarray(val_files)
        # self.x_val = x_val
        # self.y_val = y_val

    def on_epoch_end(self, epoch, logs):
        if logs is not None:
            y_pred = self.model.predict(self.validation_data[0], verbose=self.verbose)
            true = []
            pred = []
            start_idx = 0

            if self.decision == 'majority':
                y_pred = np.argmax(y_pred, axis=-1)
                y_val = np.transpose(np.argmax(self.validation_data[1], axis=-1))

                for s in self.val_parts:

                    if not s:  ## for e00032 in validation0 there was no cardiac cycle
                        continue
                    # ~ print "part {} start {} stop {}".format(s,start_idx,start_idx+int(s)-1)

                    temp_ = y_val[start_idx:start_idx + int(s) - 1]
                    temp = y_pred[start_idx:start_idx + int(s) - 1]

                    if (sum(temp == 0) > sum(temp == 1)):
                        pred.append(0)
                    else:
                        pred.append(1)

                    if (sum(temp_ == 0) > sum(temp_ == 1)):
                        true.append(0)
                    else:
                        true.append(1)

                    start_idx = start_idx + int(s)

            if self.decision =='confidence':
                y_val = np.transpose(np.argmax(self.validation_data[1], axis=-1))
                for s in self.val_parts:
                    if not s:  ## for e00032 in validation0 there was no cardiac cycle
                        continue
                    # ~ print "part {} start {} stop {}".format(s,start_idx,start_idx+int(s)-1)
                    temp_ = y_val[start_idx:start_idx + int(s) - 1]
                    if (sum(temp_ == 0) > sum(temp_ == 1)):
                        true.append(0)
                    else:
                        true.append(1)
                    temp = np.sum(y_pred[start_idx:start_idx + int(s) - 1],axis=0)
                    pred.append(int(np.argmax(temp)))
                    start_idx = start_idx + int(s)


            TN, FP, FN, TP = confusion_matrix(true, pred, labels=[0,1]).ravel()
            # TN = float(TN)
            # TP = float(TP)
            # FP = float(FP)
            # FN = float(FN)
            sensitivity = TP / (TP + FN + eps)
            specificity = TN / (TN + FP + eps)
            precision = TP / (TP + FP + eps)
            F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
            Macc = (sensitivity + specificity) / 2
            logs['val_sensitivity'] = np.array(sensitivity)
            logs['val_specificity'] = np.array(specificity)
            logs['val_precision'] = np.array(precision)
            logs['val_F1'] = np.array(F1)
            logs['val_macc'] = np.array(Macc)
            if self.verbose:
                print("TN:{},FP:{},FN:{},TP:{},Macc:{},F1:{}".format(TN, FP, FN, TP,Macc,F1))

            #### Learning Rate for Adam ###

            lr = self.model.optimizer.lr
            if self.model.optimizer.initial_decay > 0:
                lr *= (1. / (1. + self.model.optimizer.decay * K.cast(self.model.optimizer.iterations,
                                                                      K.dtype(self.model.optimizer.decay))))
            t = K.cast(self.model.optimizer.iterations, K.floatx()) + 1
            lr_t = lr * (
                    K.sqrt(1. - K.pow(self.model.optimizer.beta_2, t)) / (1. - K.pow(self.model.optimizer.beta_1, t)))
            logs['lr'] = np.array(float(K.get_value(lr_t)))

            if self.val_files is not None:
                mask = self.val_files=='x'
                TN, FP, FN, TP = confusion_matrix(np.asarray(true)[mask], np.asarray(pred)[mask], labels=[0, 1]).ravel()
                sensitivity = TP / (TP + FN + eps)
                specificity = TN / (TN + FP + eps)
                logs['ComParE_UAR'] = (sensitivity + specificity) / 2


def compute_weight(Y, classes):
    num_samples = np.float32(len(Y))
    n_classes = np.float32(len(classes))
    num_bin = np.hstack([sum(Y == 0), sum(Y == 1)])
    class_weights = {i: (num_samples / (n_classes * num_bin[i])) for i in range(2)}
    return class_weights


def reshape_folds(x_train, x_val, y_train, y_val):
    x1 = np.transpose(x_train[:, :])

    x1 = np.reshape(x1, [x1.shape[0], x1.shape[1], 1])

    y_train = np.reshape(y_train, [y_train.shape[0], 1])

    print(x1.shape)
    print(y_train.shape)

    v1 = np.transpose(x_val[:, :])

    v1 = np.reshape(v1, [v1.shape[0], v1.shape[1], 1])

    y_val = np.reshape(y_val, [y_val.shape[0], 1])

    print(v1.shape)
    print(y_val.shape)
    return x1, y_train, v1 , y_val


class show_lr(Callback):
    def on_epoch_begin(self, epoch, logs):
        print('Learning rate:')
        print(float(K.get_value(self.model.optimizer.lr)))


def lr_schedule(epoch):
    if epoch <= 5:
        lr_rate = 1e-3
    else:
        lr_rate = 1e-4 - epoch * 1e-8
    return lr_rate


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
        dropout_rate_dense = 0.2
        padding = 'valid'
        activation_function = 'relu'
        subsam = 2
        FIR_train=True
        decision = 'majority'  # Decision algorithm for inference over total recording ('majority','confidence')

        lr =  0.0012843784 ## After bayesian optimization

        ###### lr_decay optimization ######
        lr_decay =0.00001132885
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
                         padding, random_seed, subsam, num_filt, num_dense, FIR_train)
        model.summary()
        plot_model(model, to_file='model.png', show_shapes=True)

        embedding_layer_names =set(layer.name
                            for layer in model.layers
                            if (layer.name.startswith('dense_')))
        print(embedding_layer_names)
        ####### Define Callbacks ######

        modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
                                        monitor='val_acc', save_best_only=False, mode='max')
        tensbd = TensorBoard(log_dir=log_dir + log_name,
                             batch_size=batch_size, histogram_freq=100,
                             # embeddings_freq=99,
                             # embeddings_layer_names=embedding_layer_names,
                             # embeddings_data=x_val,
                             # embeddings_metadata=metadata_file,
                             write_images=False)
        csv_logger = CSVLogger(log_dir + log_name + '/training.csv')

        # show_lr()
        # log_macc()

        # Learning rate callbacks

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=lr_reduce_factor, patience=patience,
                                      min_lr=0.00001, verbose=1, cooldown=cooldown)
        dynamiclr = LearningRateScheduler(lr_schedule)
        ######### Data Generator ############

        datagen = AudioDataGenerator(shift=.1,
                                     # fill_mode='reflect',
                                     # featurewise_center=True,
                                     # zoom_range=.2,
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
                            use_multiprocessing=True,
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
        #                          log_macc(val_parts, decision=decision,verbose=verbose),
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
        #                         log_macc(val_parts, decision=decision, verbose=verbose),
        #                         tensbd, csv_logger],
        #               initial_epoch=initial_epoch)

        ############### log results in csv ###############

        df = pd.read_csv(results_path)
        df1 = pd.read_csv(log_dir + log_name + '/training.csv')
        max_idx = df1['val_macc'].idxmax()
        new_entry = {'Filename': log_name, 'Weight Initialization': 'he_normal',
                     'Activation': activation_function + '-softmax', 'Class weights': addweights,
                     'Kernel Size': kernel_size, 'Max Norm': maxnorm,
                     'Dropout -filters': dropout_rate,
                     'Dropout - dense': dropout_rate_dense,
                     'L2 - filters': l2_reg, 'L2- dense': l2_reg_dense,
                     'Batch Size': batch_size, 'Optimizer': 'Adam', 'Learning Rate': lr,
                     'BN momentum': bn_momentum, 'Lr decay': lr_decay,
                     'Best Val Acc Per Cardiac Cycle': np.mean(
                         df1.loc[max_idx - 3:max_idx + 3]['val_acc'].values) * 100,
                     'Epoch': df1.loc[[max_idx]]['epoch'].values[0],
                     'Training Acc per cardiac cycle': np.mean(df1.loc[max_idx - 3:max_idx + 3]['acc'].values) * 100,
                     'Specificity': np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_specificity'].values) * 100,
                     'Precision' : np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_precision'].values) * 100,
                     'Macc': np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_macc'].values) * 100,
                     'F1' :np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_F1'].values) * 100,
                     'Sensitivity': np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_sensitivity'].values) * 100,
                     'Number of filters': str(num_filt),
                     'Number of Dense Neurons': num_dense}

        index, _ = df.shape
        new_entry = pd.DataFrame(new_entry, index=[index])
        df2 = pd.concat([df, new_entry], axis=0)
        # df2 = df2.reindex(df.columns)
        df2.to_csv(results_path, index=False)
        df2.tail()

    except KeyboardInterrupt:
        ############ If ended in advance ###########
        df = pd.read_csv(results_path)
        df1 = pd.read_csv(log_dir + log_name + '/training.csv')
        max_idx = df1['val_macc'].idxmax()
        new_entry = {'Filename': '*' + log_name, 'Weight Initialization': 'he_normal',
                     'Activation': activation_function + '-softmax', 'Class weights': addweights,
                     'Kernel Size': kernel_size, 'Max Norm': maxnorm,
                     'Dropout -filters': dropout_rate,
                     'Dropout - dense': dropout_rate_dense,
                     'L2 - filters': l2_reg, 'L2- dense': l2_reg_dense,
                     'Batch Size': batch_size, 'Optimizer': 'Adam', 'Learning Rate': lr,
                     'BN momentum': bn_momentum,'Lr decay': lr_decay,
                     'Best Val Acc Per Cardiac Cycle': np.mean(
                         df1.loc[max_idx - 3:max_idx + 3]['val_acc'].values) * 100,
                     'Epoch': df1.loc[[max_idx]]['epoch'].values[0],
                     'Training Acc per cardiac cycle': np.mean(df1.loc[max_idx - 3:max_idx + 3]['acc'].values) * 100,
                     'Specificity': np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_specificity'].values) * 100,
                     'Macc': np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_macc'].values) * 100,
                     'Precision': np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_precision'].values) * 100,
                     'Sensitivity': np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_sensitivity'].values) * 100,
                     'Number of filters': str(num_filt),
                     'F1': np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_F1'].values) * 100,
                     'Number of Dense Neurons': num_dense}

        index, _ = df.shape
        new_entry = pd.DataFrame(new_entry, index=[index])
        df2 = pd.concat([df, new_entry], axis=0)
        # df2 = df2.reindex(df.columns)
        df2.to_csv(results_path, index=False)
        df2.tail()
        print("Saving to results.csv")

