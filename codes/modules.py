import os
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Activation
from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.models import Model
from learnableFilterbanks import Conv1D_zerophase,Conv1D_gammatone, Conv1D_linearphaseType
from utils import loadFIRparams


def branch(input_tensor,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable):
    """
    Branched CNN architecture
    :return: Keras layer object
    """

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
    return t


def heartnetTop(activation_function='relu', bn_momentum=0.99, bias=False, dropout_rate=0.5,
             eps=1.1e-5, kernel_size=5, l2_reg=0.0, maxnorm=10000.,
             padding='valid', random_seed=1, subsam=2, num_filt=(8, 4), FIR_train=False,trainable=True,FIR_type=1):
    """
    Heartnet Topmodel (without dense layers)
    :return: Keras model object
    """

    input = Input(shape=(2500, 1))

    coeff_path = os.path.join('..','data','filterbankInit','filterbankcoeff60.mat')
    b1,b2,b3,b4 = loadFIRparams(coeff_path)

    if type(FIR_type) == str and FIR_type == 'gamma':

        input1 = Conv1D_gammatone(kernel_size=81,filters=1,fsHz=1000,use_bias=False,padding='same')(input)
        input2 = Conv1D_gammatone(kernel_size=81,filters=1,fsHz=1000,use_bias=False,padding='same')(input)
        input3 = Conv1D_gammatone(kernel_size=81,filters=1,fsHz=1000,use_bias=False,padding='same')(input)
        input4 = Conv1D_gammatone(kernel_size=81,filters=1,fsHz=1000,use_bias=False,padding='same')(input)

    elif type(FIR_type) == str and FIR_type == 'zero':

        input1 = Conv1D_zerophase(1 ,61, use_bias=False,
                        weights=[b1],
                        padding='same',trainable=FIR_train)(input)
        input2 = Conv1D_zerophase(1, 61, use_bias=False,
                        weights=[b2],
                        padding='same',trainable=FIR_train)(input)
        input3 = Conv1D_zerophase(1, 61, use_bias=False,
                        weights=[b3],
                        padding='same',trainable=FIR_train)(input)
        input4 = Conv1D_zerophase(1, 61, use_bias=False,
                        weights=[b4],
                        padding='same',trainable=FIR_train)(input)

    ### Linearphase
    else:
        FIR_type = int(FIR_type)
        if FIR_type % 2:
            weight_idx = 30
        else:
            weight_idx = 31

        input1 = Conv1D_linearphaseType(1 ,61, use_bias=False,
                        weights=[b1[weight_idx:]],
                        padding='same',trainable=FIR_train, FIR_type = FIR_type)(input)
        input2 = Conv1D_linearphaseType(1, 61, use_bias=False,
                        weights=[b2[weight_idx:]],
                        padding='same',trainable=FIR_train, FIR_type = FIR_type)(input)
        input3 = Conv1D_linearphaseType(1, 61, use_bias=False,
                        weights=[b3[weight_idx:]],
                        padding='same',trainable=FIR_train, FIR_type = FIR_type)(input)
        input4 = Conv1D_linearphaseType(1, 61, use_bias=False,
                        weights=[b4[weight_idx:]],
                        padding='same',trainable=FIR_train, FIR_type = FIR_type)(input)

    t1 = branch(input1,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t2 = branch(input2,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t3 = branch(input3,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t4 = branch(input4,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)

    output = Concatenate(axis=-1)([t1, t2, t3, t4])
    model = Model(input,output)
    return model