import tables
import numpy as np
import os
from keras.utils import to_categorical

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