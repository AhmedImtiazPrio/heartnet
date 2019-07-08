import tables
import numpy as np
import os
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.callbacks import Callback
from keras import backend as K

class log_macc_LEGACY(Callback):

    def __init__(self, val_parts,decision='majority',verbose=0, val_files=None):
        super(log_macc_LEGACY, self).__init__()
        self.val_parts = val_parts
        self.decision = decision
        self.verbose = verbose
        self.val_files = np.asarray(val_files)
        # self.x_val = x_val
        # self.y_val = y_val

    def on_epoch_end(self, epoch, logs):
        eps = 1.1e-5
        if logs is not None:
            y_pred = self.model.predict(self.validation_data[0], verbose=self.verbose)
            true = []
            pred = []
            files= []
            start_idx = 0

            if self.decision == 'majority':
                y_pred = np.argmax(y_pred, axis=-1)
                y_val = np.transpose(np.argmax(self.validation_data[1], axis=-1))

                for s in self.val_parts:

                    if not s:  ## for e00032 in validation0 there was no cardiac cycle
                        continue
                    # ~ print "part {} start {} stop {}".format(s,start_idx,start_idx+int(s)-1)

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

                    if self.val_files is not None:
                        files.append(self.val_files[start_idx])

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
            F1 = 2 * (precision * sensitivity) / (precision + sensitivity + eps)
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
                true = np.asarray(true)
                pred = np.asarray(pred)
                files = np.asarray(files)
                tpn = true == pred
                for dataset in set(files):
                    mask = files == dataset
                    logs['acc_'+dataset] = np.sum(tpn[mask])/np.sum(mask)
                    # mask = self.val_files=='x'
                    # TN, FP, FN, TP = confusion_matrix(np.asarray(true)[mask], np.asarray(pred)[mask], labels=[0, 1]).ravel()
                    # sensitivity = TP / (TP + FN + eps)
                    # specificity = TN / (TN + FP + eps)
                    # logs['ComParE_UAR'] = (sensitivity + specificity) / 2


def load_data_LEGACY(HS, data_dir, _categorical=True, quality=False):
    """
    Helper function to load HS data
    :param HS: data tensor
    :param data_dir: data directory
    :param _categorical: {True,False} if true labels to categorical
    :param quality: {True,False} if true, also returns recording quality
    :return: x_train, y_train, train_subset, train_parts, x_val, y_val, val_subset, val_parts
    """
    feat = tables.open_file(os.path.join(data_dir,HS+'.mat'))
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

    train_subset = []
    for each in feat.root.train_files[:][0]:
        train_subset.append(chr(each))
    print(len(train_subset))
    val_subset = []
    for each in feat.root.val_files[:][0]:
        val_subset.append(chr(each))
    print(len(val_subset))

    ################### Reshaping ############

    x_train, y_train, x_val, y_val = reshape_folds_LEGACY(x_train, x_val, y_train, y_val)

    if _categorical:
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)

    if quality:
        return x_train, y_train, train_subset, train_parts, q_train, \
               x_val, y_val, val_subset, val_parts, q_val
    else:
        return x_train, y_train, train_subset, train_parts, \
               x_val, y_val, val_subset, val_parts

def reshape_folds_LEGACY(x_train, x_val, y_train, y_val):
    x1 = np.transpose(x_train[:, :])
    x1 = np.reshape(x1, [x1.shape[0], x1.shape[1], 1])
    y_train = np.reshape(y_train, [y_train.shape[0], 1])
    v1 = np.transpose(x_val[:, :])
    v1 = np.reshape(v1, [v1.shape[0], v1.shape[1], 1])
    y_val = np.reshape(y_val, [y_val.shape[0], 1])
    return x1, y_train, v1 , y_val