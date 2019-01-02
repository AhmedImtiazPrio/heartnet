from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers import Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

def conv_factory(x, concat_axis=-1, nb_filter=16,
                 dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv1D, optional dropout
    :param x: Input keras network
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and Conv1D added
    :rtype: keras network
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv1D(nb_filter, 4,
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, concat_axis, nb_filter,
               dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv1D, optional dropout and Maxpooling2D
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv1D(nb_filter, 4,
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling1D(2, strides=2)(x)

    return x


def denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """

    list_feat = [x]

    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)
        nb_filter += growth_rate

    return x, nb_filter


def denseblock_altern(x, concat_axis, nb_layers, nb_filter, growth_rate,
                      dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each conv_factory
       is fed to subsequent ones. (Alternative of a above)
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    * The main difference between this implementation and the implementation
    above is that the one above
    """

    for i in range(nb_layers):
        merge_tensor = conv_factory(x, concat_axis, growth_rate,
                                    dropout_rate, weight_decay)
        x = Concatenate(axis=concat_axis)([merge_tensor, x])
        nb_filter += growth_rate

    return x, nb_filter
