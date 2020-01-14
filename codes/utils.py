from keras.callbacks import Callback
import os
import pandas as pd
import numpy as np
from keras import backend as K
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from learnableFilterbanks import Conv1D_linearphase,DCT1D,Conv1D_linearphaseType,\
    Conv1D_gammatone,Conv1D_zerophase,Conv1D_linearphaseType_legacy
from keras.utils import to_categorical
from keras.models import model_from_json
import tables
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy import signal
import csv
from collections import Counter

def sessionLog(results_path,log_dir,log_name,activation_function,kernel_size,maxnorm,
                dropout_rate,dropout_rate_dense,l2_reg,l2_reg_dense,batch_size,lr,bn_momentum,
                lr_decay,num_dense,comment,num_filt,initializer,optimizer,verbose,random_seed,subsam,**kwargs):
    """
    Session logger helper function
    """

    keys = [
        'logname','random_seed','weightInit','activation','kernelSize','subsam','numFilt','maxNorm','dropoutFilt','numDense','dropoutDense',
        'l2Filt','l2Dense','batchSize','optimizer','lr','bnMomentum','lrDecay','epoch',
        'trainCCacc','valCCacc','sens','spec','macc','prec','F1','comment'
              ]

    if os.path.isfile(results_path):
        df = pd.read_csv(results_path)
    else:
        with open(results_path,'w') as writeFile:
            writer = csv.writer(writeFile, lineterminator='\n')
            writer.writerow(keys)
        df = pd.read_csv(results_path)

    dfNew = pd.read_csv(os.path.join(log_dir,log_name,'training.csv'))
    max_idx = dfNew['val_macc'].idxmax()

    epoch = dfNew.loc[[max_idx]]['epoch'].values[0]
    trainAcc = dfNew.loc[max_idx]['acc'] * 100
    valAcc = dfNew.loc[max_idx]['val_acc'] * 100
    sens = dfNew.loc[max_idx]['val_sensitivity'] * 100
    spec = dfNew.loc[max_idx]['val_specificity'] * 100
    macc = dfNew.loc[max_idx]['val_macc'] * 100
    prec = dfNew.loc[max_idx]['val_precision'] * 100
    F1 = dfNew.loc[max_idx]['val_F1'] * 100

    values = [
        log_name,random_seed,initializer.__name__,activation_function,kernel_size,subsam,str(num_filt),maxnorm,dropout_rate,num_dense,dropout_rate_dense,
        l2_reg,l2_reg_dense,batch_size,optimizer.__name__,lr,bn_momentum,lr_decay,epoch,trainAcc,valAcc,sens,spec,
        macc,prec,F1,comment
             ]
    new_entry = dict(zip(keys,values))
    index, _ = df.shape
    new_entry = pd.DataFrame(new_entry,index=[index])
    df = pd.concat([df, new_entry], axis=0)
    df.to_csv(results_path, index=False)

    if verbose:
        df.tail()

    print("Saving to results.csv")

class log_metrics(Callback):
    '''
    Keras Callback for custom metric logging
    '''

    def __init__(self, val_parts, val_subset=None, soft=False, verbose=0 ):
        super(log_metrics, self).__init__()
        self.val_parts = val_parts
        if val_subset is not None:
            self.val_subset = np.asarray(val_subset)
        self.soft = soft
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs):
        eps = 1.1e-5
        if logs is not None:
            true,pred,subset = predict_parts(self.model,self.validation_data[0],
                                                self.validation_data[1],self.val_parts,
                                                self.val_subset,self.verbose,self.soft)

            metrics = calc_metrics(true,pred,subset,verbose=1,thresh=.5,outputDict=True)
            logs.update(metrics)

            #### Learning Rate for Adam ###

            lr = self.model.optimizer.lr
            if self.model.optimizer.initial_decay > 0:
                lr *= (1. / (1. + self.model.optimizer.decay * K.cast(self.model.optimizer.iterations,
                                                                      K.dtype(self.model.optimizer.decay))))
            t = K.cast(self.model.optimizer.iterations, K.floatx()) + 1
            lr_t = lr * (
                    K.sqrt(1. - K.pow(self.model.optimizer.beta_2, t)) / (1. - K.pow(self.model.optimizer.beta_1, t)))
            logs['lr'] = np.array(float(K.get_value(lr_t)))


class LRdecayScheduler(Callback):
    def __init__(self, schedule, verbose=0):
        super(LRdecayScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'decay'):
            raise ValueError('Optimizer must have a "decay" attribute.')
        lr_decay = float(K.get_value(self.model.optimizer.decay))
        try:  # new API
            lr_decay = self.schedule(epoch, lr_decay)
        except TypeError:  # old API for backward compatibility
            lr_decay = self.schedule(epoch)
        if not isinstance(lr_decay, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if lr_decay > 0.:
            K.set_value(self.model.optimizer.decay, lr_decay)
            self.model.optimizer.initial_decay =  lr_decay
        if self.verbose > 0:
            print('\nEpoch %05d: LRdecayScheduler setting decay '
                  'rate to %s.' % (epoch + 1, lr_decay))


def loadFIRparams(coeff_path):
    """
    Utility function to load filterbank parameter matfile
    :param path to load filterbank parameters from:
    :return: filterbank parameters
    """
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
    return b1,b2,b3,b4


def get_activations(model, model_inputs, batch_size=64, print_shape_only=True, layer_name=None):
    '''
    Get activations from a specific layer of a trained model
    '''
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    start_idx = 0
    for idx in range(batch_size, len(model_inputs), batch_size):
        print(batch_size)
        if model_multi_inputs_cond:
            raise NotImplementedError
        else:
            list_inputs = [model_inputs[start_idx:idx], 0.]

        # Learning phase. 0 = Test mode (no dropout or batch normalization)
        # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
        layer_outputs = [func(list_inputs)[0] for func in funcs]
        for layer_activations in layer_outputs:
            activations.append(layer_activations)
        start_idx = idx
    return np.vstack(activations)


def display_activations(activation_maps):
    '''
    Plot activations
    '''
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.imshow(activations, interpolation='None', cmap='jet')
    plt.show()


def smooth(scalars, weight):
    '''
    Smoothing function
    :param scalars: list of scalars
    :param weight: between 0 and 1
    :return: smoothed output
    '''
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return np.asarray(smoothed)


def get_weights(log_name, min_metric=.7, min_epoch=50, verbose=1, log_dir='/media/taufiq/Data1/heart_sound/logs'):
    '''
    Load weights from training.csv file
    '''
    #     log_dir = '/media/taufiq/Data1/heart_sound/logs'

    if not os.path.isdir(os.path.join(log_dir, log_name)):
        log_dir = '/media/taufiq/Data1/heart_sound/logArxiv'
    training_csv = os.path.join(log_dir, log_name, "training.csv")
    df = pd.read_csv(training_csv)
    sens_idx = df['val_sensitivity'][df.epoch > min_epoch][df.val_specificity > min_metric].idxmax()
    spec_idx = df['val_specificity'][df.epoch > min_epoch][df.val_sensitivity > min_metric].idxmax()
    macc_idx = df['val_macc'][df.epoch > min_epoch].idxmax()
    val_idx = df['val_acc'][df.epoch > min_epoch].idxmax()
    weights = dict()
    weights['val_sensitivity'] = "weights.%.4d-%.4f.hdf5" % (df.epoch.iloc[sens_idx] + 1, df.val_acc.iloc[sens_idx])
    weights['val_specificity'] = "weights.%.4d-%.4f.hdf5" % (df.epoch.iloc[spec_idx] + 1, df.val_acc.iloc[spec_idx])
    weights['val_macc'] = "weights.%.4d-%.4f.hdf5" % (df.epoch.iloc[macc_idx] + 1, df.val_acc.iloc[macc_idx])
    weights['val_acc'] = "weights.%.4d-%.4f.hdf5" % (df.epoch.iloc[val_idx] + 1, df.val_acc.iloc[val_idx])
    if verbose:
        print("Best Sensitivity model: {} \t\t{}".format(df.val_sensitivity.iloc[sens_idx], weights['val_sensitivity']))
        print("Best Specificity model: {} \t\t{}".format(df.val_specificity.iloc[spec_idx], weights['val_specificity']))
        print("Best Macc model: {} \t\t{}".format(df.val_macc.iloc[macc_idx], weights['val_macc']))
        print("Best Val model: {} \t\t\t{}".format(df.val_acc.iloc[val_idx], weights['val_acc']))
    return weights


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

     
def load_data(foldname,fold_dir=None,_categorical=True,quality=False):
    ## import data
    if fold_dir is None:
        fold_dir = '/media/mhealthra2/Data/heart_sound/feature/segmented_noFIR/folds_dec_2018/'
    else:
        print(os.path.join(fold_dir,foldname+'.mat'))
    feat = tables.open_file(os.path.join(fold_dir,foldname + '.mat'))
    x_train = feat.root.trainX[:]
    y_train = feat.root.trainY[0, :]
    q_train = feat.root.trainY[1, :]
    x_val = feat.root.valX[:]
    y_val = feat.root.valY[0, :]
    q_val = feat.root.valY[1, :]
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

    if _categorical:
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)
    
    if quality:
        return x_train, y_train, train_files, train_parts, q_train, \
                x_val, y_val, val_files, val_parts, q_val
    else:
        return x_train, y_train, train_files, train_parts, \
                x_val, y_val, val_files, val_parts

def load_dataa(HS, data_dir='../data/feature/folds', _categorical=True, quality=False):
    """
    Helper function to load HS data
    :param HS: data tensor
    :param data_dir: data directory
    :param _categorical: {True,False} if true labels to categorical
    :param quality: {True,False} if true, also returns recording quality
    :return: x_train, y_train, train_subset, train_parts, x_val, y_val, val_subset, val_parts
    """
    print(os.path.join(data_dir,HS+'.mat'))
    feat = loadmat(os.path.join(data_dir,HS+'.mat'))
    x_train = feat['trainX']
    y_train = feat['trainY'][:,0]
    q_train = feat['trainY'][:,1]
    x_val = feat['valX']
    y_val = feat['valY'][:,0]
    q_val = feat['valY'][:,1]
    train_parts = feat['trainParts'][:,0]
    val_parts = feat['valParts'][:,0]

    ############## Relabeling ################

    y_train = np.asarray([int(each>0)for each in y_train])
    y_val = np.asarray([int(each>0)for each in y_val])

    ############# Parse Filenames ########

    train_files = np.asarray([each[0][0] for each in feat['trainFiles']])
    val_files = np.asarray([each[0][0] for each in feat['valFiles']])

    ################### Reshaping ############

    x_train, y_train, x_val, y_val = reshape_folds(x_train, x_val, y_train, y_val)

    if _categorical:
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)

    if quality:
        return x_train, y_train, train_files, train_parts, q_train, \
               x_val, y_val, val_files, val_parts, q_val
    else:
        return x_train, y_train, train_files, train_parts, \
               x_val, y_val, val_files, val_parts


def load_model(log_name, verbose=0,
               model_dir='/media/taufiq/Data1/heart_sound/models/',
               log_dir='/media/taufiq/Data1/heart_sound/logs/'):
    #     model_dir = '/media/taufiq/Data1/heart_sound/models/'
    #     log_dir = '/media/taufiq/Data1/heart_sound/logs/'

    if os.path.isdir(model_dir + log_name):
        print("Model directory found")
        if os.path.isfile(os.path.join(model_dir + log_name, "model.json")):
            print("model.json found. Importing")
        else:
            raise ImportError("model.json not found")

    with open(os.path.join(model_dir + log_name, "model.json")) as json_file:
        loaded_model_json = json_file.read()
    try:
        model = model_from_json(loaded_model_json, {'Conv1D_linearphase': Conv1D_linearphase,
                                                    'DCT1D': DCT1D,
                                                    'Conv1D_linearphaseType': Conv1D_linearphaseType,
                                                    'Conv1D_gammatone': Conv1D_gammatone,
                                                    'Conv1D_zerophase': Conv1D_zerophase,
                                                    })
    except:
        model = model_from_json(loaded_model_json, {'Conv1D_linearphase': Conv1D_linearphase,
                                                    'DCT1D': DCT1D,
                                                    'Conv1D_linearphaseType': Conv1D_linearphaseType_legacy,
                                                    'Conv1D_gammatone': Conv1D_gammatone,
                                                    'Conv1D_zerophase': Conv1D_zerophase,
                                                    })

    if verbose:
        print(log_name)
        model.summary()
    return model


def parts2rec(cc, parts):
    '''
    Take labels from cardiac cycle level to recording level
    :param cc: cardiac cycle level tensor
    :param parts: vector conatining number of cc for each recording
    :return: recording level tensor
    '''
    if not len(cc) == sum(parts):
        raise ValueError('Number of CC elements are not equal to total number of parts')

    labels = []
    start_idx = 0
    #     cc = np.round(cc)

    for s in parts:
        if not s:  ## for e00032 in validation0 there was no cardiac cycle
            continue
        temp = cc[start_idx:start_idx + int(s)]
        try:
            labels.append(np.mean(temp, axis=0))
        except TypeError:  ## TypeError for string input in train_files
            labels.append(cc[start_idx])
        start_idx = start_idx + int(s)
    return np.asarray(labels)


def rec2parts(partitioned, parts):
    labels = []
    parts = parts[np.nonzero(parts)]
    for each, part in zip(partitioned, parts):
        labels += list(np.repeat(each, part))
    return np.asarray(labels)


def predict_parts(model, data, labels, parts, filenames=None, verbose=1, soft=False):
    '''
    Helper function to get predictions for each recording
    :param soft: True for softscore fusion, False for majority voting
    :return: recording level prediction, ground truth and filenames
    '''
    y_pred = model.predict(data, verbose=verbose)
    true = []
    pred = []
    files = []
    start_idx = 0
    y_pred = np.argmax(y_pred, axis=-1)
    y_val = np.transpose(np.argmax(labels, axis=-1))
    for s in parts:
        if not s:  ## for e00032 in validation0 there was no cardiac cycle
            continue
        # print("part {} start {} stop {}".format(s,start_idx,start_idx+int(s)-1))
        # print(filenames[start_idx:start_idx+s])
        temp_ = y_val[start_idx:start_idx + int(s)]
        temp = y_pred[start_idx:start_idx + int(s)]
        if (sum(temp == 0) > sum(temp == 1)):
            pred.append(0)
        else:
            pred.append(1)

        if (sum(temp_ == 0) > sum(temp_ == 1)):
            true.append(0)
        else:
            true.append(1)

        if filenames is not None:
            files.append(filenames[start_idx])
        start_idx = start_idx + int(s)

    if soft:
        pred = parts2rec(y_pred, parts)
    return pred, true, files

def eerPred(true,pred,verbose=1):
    '''
    Calculate equal error rate predictions
    '''
    if pred.ndim > 1:
            pred = pred[:,-1]
    fpr,tpr,thresh = roc_curve(true,pred)
    diff = abs(tpr-(1-fpr))
    pred = pred > thresh[np.where(diff == min(diff))[0]]
    if verbose:
        print('Threshold selected as %f'%thresh[np.where(diff == min(diff))[0]])
    return pred


def calc_metrics(true,pred,subset=None,verbose=1,eps=1E-10,thresh=.5,outputDict=False):
    '''
    Calculates sens, spec, prec, F1, Macc, MCC and auc metrics
    :param true: ground truth
    :param pred: predictions
    :param subset: subset names
    :param verbose: verbosity
    :param eps: epsilon
    :param thresh: decision threshold
    :return: pandas dataframe with
    '''
    if thresh=='EER':
        TN, FP, FN, TP = confusion_matrix(true, eerPred(true,pred), labels=[0,1]).ravel()
    else:
        TN, FP, FN, TP = confusion_matrix(true, np.asarray(pred) > thresh, labels=[0,1]).ravel()
    sensitivity = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    precision = TP / (TP + FP + eps)
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity + eps)
    Macc = (sensitivity + specificity) / 2
    MCC = (TP*TN-FP*FN)/((TP+FP)*(FN+TN)*(FP+TN)*(TP+FN))**.5
    auc = roc_auc_score(true,pred)
    logs = dict()
    logs['val_sensitivity'] = np.array(sensitivity)
    logs['val_specificity'] = np.array(specificity)
    logs['val_precision'] = np.array(precision)
    logs['val_F1'] = np.array(F1)
    logs['val_macc'] = np.array(Macc)
    logs['auc'] = np.array(auc)
    logs['val_mcc'] = np.array(MCC).astype(np.float64)
    if verbose:
        print("TN:{},FP:{},FN:{},TP:{},Macc:{},F1:{}".format(TN, FP, FN, TP,Macc,F1))
    if subset is not None:
        true = np.asarray(true)
        pred = np.asarray(pred) > .5
        subset = np.asarray(subset)
        tpn = true == pred
        avg = 0
        for dataset in np.unique(subset):
            mask = subset == dataset
            avg = avg + np.sum(tpn[mask])/np.sum(mask)/len(np.unique(subset))
            logs['acc_'+dataset] = np.sum(tpn[mask])/np.sum(mask)
        logs['acc_avg'] = avg
    if outputDict:
        return logs
    else:
        df = pd.Series(logs)
        return df


def log_fusion(logs, data, labels, fusion_weights=None, min_epoch=20, min_metric=.7,
               metric='val_macc', model_dir='/media/taufiq/Data1/heart_sound/models/', verbose=0):
    '''
    Returns fused predictions
    '''
    if not type(logs) == list:
        logs = [logs]

    if fusion_weights is None:
        fusion_weights = np.ones((len(logs)))
    else:
        if not len(logs) == len(fusion_weights):
            raise ValueError('Fusion weights not consistent with number of models')
    pred = np.zeros((data.shape[0], 2))

    for log_name, weight in zip(logs, fusion_weights):
        model = load_model(log_name, verbose=verbose)
        weights = get_weights(log_name, min_epoch=min_epoch,
                              min_metric=min_metric, verbose=verbose)
        checkpoint_name = os.path.join(model_dir + log_name, weights[metric])
        model.load_weights(checkpoint_name)
        pred += model.predict(data, verbose=verbose) * weight
    pred /= sum(fusion_weights)
    # pred = np.argmax(pred,axis=-1)
    return pred


def model_confidence(model, data, labels, verbose=0):
    '''
    Give confidence score for true class
    '''
    pred = model.predict(data, verbose=verbose)

    if np.asarray(labels).ndim > 1:
        labels = np.argmax(labels, axis=-1)

    pred = [pred[idx, each] for idx, each in enumerate(labels)]

    return np.asarray(pred)


def plot_coeff(logs, branches=[1, 2, 3, 4], min_epoch=20, min_metric=.7,
               metric='val_macc', model_dir='/media/taufiq/Data1/heart_sound/models/',
               figsize=(10, 6), verbose=0):
    '''
    Plot Learnable FIRs for logs
    '''
    if not type(logs) == list:
        logs = [logs]
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(len(branches), len(logs), sharex='col', sharey='row', figsize=figsize)

    for _idx, log_name in enumerate(logs):
        model = load_model(log_name, verbose=verbose)
        weights = get_weights(log_name, min_epoch=min_epoch,
                              min_metric=min_metric, verbose=verbose)
        checkpoint_name = os.path.join(model_dir + log_name, weights[metric])
        model.load_weights(checkpoint_name)

        FIR_coeff = []
        layer_name = []
        layer_type = []

        ## Get filter coefficients
        for branch in branches:
            if not 'gammatone' in model.layers[branch].name:
                FIR_coeff.append(np.asarray(model.layers[branch].get_weights())[0, :, 0, 0])
                layer_name.append(model.layers[branch].name)
            else:  # for gammatone
                FIR_coeff.append(K.get_session().run(model.layers[branch].impulse_gammatone()))
                layer_name.append(model.layers[branch].name)
            try:
                layer_type.append(model.layers[branch].FIR_type)
            except:  # if not linear phase
                layer_type.append(0)

        for idx, coeff in enumerate(FIR_coeff):

            ## Flip-concat coefficients for Linearphase
            if 'linearphase' in layer_name[idx]:
                if layer_type[idx] == 1:
                    FIR_coeff[idx] = np.concatenate([np.flip(FIR_coeff[idx][1:], axis=0), FIR_coeff[idx]])
                elif layer_type[idx] == 2:
                    FIR_coeff[idx] = np.concatenate([np.flip(FIR_coeff[idx], axis=0), FIR_coeff[idx]])
                elif layer_type[idx] == 3:
                    FIR_coeff[idx] = np.concatenate([-1 * np.flip(FIR_coeff[idx][1:], axis=0), FIR_coeff[idx]])
                else:
                    FIR_coeff[idx] = np.concatenate([-1 * np.flip(FIR_coeff[idx], axis=0), FIR_coeff[idx]])

            ax[idx, _idx].plot((FIR_coeff[idx] - np.mean(FIR_coeff[idx])) / np.std(FIR_coeff[idx]))

    plt.tight_layout()
    plt.show()
    return ax


def plot_freq(logs, branches=[1, 2, 3, 4], phase=False, min_epoch=20, min_metric=.7,
              metric='val_macc', model_dir='/media/taufiq/Data1/heart_sound/models/',
              figsize=(10, 6), verbose=0):
    '''
    Plot Learnable FIRs for logs
    '''
    if not type(logs) == list:
        logs = [logs]
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(len(branches), len(logs), sharex='col', sharey='row', figsize=figsize)

    for _idx, log_name in enumerate(logs):
        model = load_model(log_name, verbose=verbose)
        weights = get_weights(log_name, min_epoch=min_epoch,
                              min_metric=min_metric, verbose=verbose)
        checkpoint_name = os.path.join(model_dir + log_name, weights[metric])
        model.load_weights(checkpoint_name)

        FIR_coeff = []
        layer_name = []
        layer_type = []

        ## Get filter coefficients
        for branch in branches:
            if not 'gammatone' in model.layers[branch].name:
                FIR_coeff.append(np.asarray(model.layers[branch].get_weights())[0, :, 0, 0])
                layer_name.append(model.layers[branch].name)
            else:  # for gammatone
                FIR_coeff.append(K.get_session().run(model.layers[branch].impulse_gammatone()))
                layer_name.append(model.layers[branch].name)
            try:
                layer_type.append(model.layers[branch].FIR_type)
            except:  # if not linear phase
                layer_type.append(0)

        for idx, coeff in enumerate(FIR_coeff):

            ## Flip-concat coefficients for Linearphase
            if 'linearphase' in layer_name[idx]:
                if layer_type[idx] == 1:
                    FIR_coeff[idx] = np.concatenate([np.flip(FIR_coeff[idx][1:], axis=0), FIR_coeff[idx]])
                elif layer_type[idx] == 2:
                    FIR_coeff[idx] = np.concatenate([np.flip(FIR_coeff[idx], axis=0), FIR_coeff[idx]])
                elif layer_type[idx] == 3:
                    FIR_coeff[idx] = np.concatenate([-1 * np.flip(FIR_coeff[idx][1:], axis=0), FIR_coeff[idx]])
                else:
                    FIR_coeff[idx] = np.concatenate([-1 * np.flip(FIR_coeff[idx], axis=0), FIR_coeff[idx]])

            w, freq_res = signal.freqz(FIR_coeff[idx])
            ax[idx, _idx].plot(w / np.pi * 500, 10 * np.log10(abs(freq_res) / max(abs(freq_res))))
            if phase:
                angles = np.unwrap(np.angle(freq_res))
                ax2 = ax[idx, _idx].twinx()
                ax2.plot(w / np.pi * 500, angles, 'g')

    plt.tight_layout()
    #     plt.show()
    return ax


def plot_metric(logs, metric='val_loss', smoothing=0.1, lognames=None, xlim=None, ylim=None,
                figsize=(10, 6), legendLoc=0, colors=None, ax=None):
    '''
    Plot specified metric for logs
    smooth: smoothing factor for each plot
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for idx, log in enumerate(logs):
        log_dir = '/media/taufiq/Data1/heart_sound/logs'
        if not os.path.isdir(os.path.join(log_dir, log)):
            log_dir = '/media/taufiq/Data1/heart_sound/logArxiv'
        training_csv = os.path.join(log_dir, log, "training.csv")
        df = pd.read_csv(training_csv)
        data = np.asarray(df[metric].values)

        if colors is not None:
            ax.plot(smooth(data, smoothing), color=colors[idx])
        else:
            ax.plot(smooth(data, smoothing))

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if lognames is not None:
        ax.legend(lognames, loc=legendLoc)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric)

    return ax


def plot_log_metrics(log, metrics=['acc_a', 'acc_e'], labels=None, smoothing=0.1,
                     xlim=None, ylim=None, figsize=(10, 6), legendLoc=0, colors=None, ax=None):
    '''
    Plot multiple metrics of the same log
    '''
    log_dir = '/media/taufiq/Data1/heart_sound/logs'
    if not os.path.isdir(os.path.join(log_dir, log)):
        log_dir = '/media/taufiq/Data1/heart_sound/logArxiv'
    training_csv = os.path.join(log_dir, log, "training.csv")
    df = pd.read_csv(training_csv)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for idx, metric in enumerate(metrics):
        data = np.asarray(df[metric].values)
        if colors is not None:
            ax.plot(smooth(data, smoothing), color=colors[idx])
        else:
            ax.plot(smooth(data, smoothing))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    #     if lognames is not None:
    #         ax.legend(metrics,loc=legendLoc)
    ax.set_xlabel('Epochs')
    return ax


def idx_rec2parts(partidx, parts):
    if type(partidx) == int:
        partidx = [partidx]

    idx = []
    for each in partidx:
        start_idx = int(sum(parts[:each]))
        end_idx = int(start_idx + parts[each])
        idx = idx + range(start_idx, end_idx)
    return idx


def smooth_win(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]


def grad_cam(model, layer_name, data, label, scale=True, verbose=0):
    if data.ndim < 3:
        data = np.expand_dims(data, axis=0)
    output = model.output[:, 1 - int(label)]
    last_conv_layer = model.get_layer(layer_name)  ##### have to change the name here
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1))  ### no idea what to do here
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([data])
    for i in range(pooled_grads_value.shape[0]):
        if verbose:
            print("Iteration %d" % i)
        conv_layer_output_value[:, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    if scale:
        x = np.linspace(0, data.shape[1], num=len(heatmap))
        y = heatmap
        f1 = interp1d(x, y, kind='cubic')
        xnew = np.linspace(0, data.shape[1], num=data.shape[1])
        ynew = f1(xnew)
        return ynew
    else:
        return heatmap


def cc2rec(data):
    rec = []
    for cc in data:
        idx = np.where(cc != 0)[0]
        cc = cc[:idx[-1], 0]
        rec.append(cc)
    return np.asarray(np.hstack(rec))


def cc2rec_labels(data, labels):
    gt = []
    for i, cc in enumerate(data):
        idx = np.where(cc != 0)[0]
        cctr = np.ones(idx[-1]) * labels[i]
        gt.append(cctr)
    return np.asarray(np.hstack(gt))
