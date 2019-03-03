from __future__ import print_function, absolute_import, division
from keras import backend as K
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
import tensorflow as tf
from keras.utils import conv_utils
from keras.layers import activations, initializers, regularizers, constraints
import numpy as np
import warnings
# from scipy.fftpack import dct


class Conv1D_zerophase(Layer):

    def __init__(self, filters, kernel_size, rank=1, strides=1, padding='valid', data_format='channels_last',
                 dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super(Conv1D_zerophase, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        outputs = K.conv1d(
            inputs,
            self.kernel,
            strides=self.strides[0],
            padding='same',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])
        outputs = tf.reverse(outputs, axis=[1])
        outputs = K.conv1d(
            outputs,
            self.kernel,
            strides=self.strides[0],
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])
        outputs = tf.reverse(outputs, axis=[1])
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        kernel_size = list(self.kernel_size)
        if self.kernel_size[0] % 2:
            kernel_size[0] = kernel_size[0]*2 -1
        else:
            kernel_size[0] = kernel_size[0]*2
        kernel_size = tuple(kernel_size)
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv1D_zerophase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Conv1D_zerophase_linear(Layer):

    def __init__(self, filters, kernel_size, rank=1, strides=1, padding='valid', data_format='channels_last',
                 dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super(Conv1D_zerophase_linear, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size_ = kernel_size
        if kernel_size % 2:
            self.kernel_size = conv_utils.normalize_tuple(kernel_size // 2 + 1, rank, 'kernel_size')
        else:
            self.kernel_size = conv_utils.normalize_tuple(kernel_size // 2, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        if self.kernel_size[0] % 2 == 0:
            flipped = tf.reverse(self.kernel, axis=[0])
        else:
            flipped = tf.reverse(self.kernel[1:, :, :], axis=[0])
        #         print (flipped)
        conv_kernel = tf.concat([flipped, self.kernel], axis=0)
        #         print (conv_kernel)

        outputs = K.conv1d(
            inputs,
            conv_kernel,
            strides=self.strides[0],
            padding='same',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])
        #         print tf.shape(outputs)
        outputs = tf.reverse(outputs, axis=[1])
        outputs = K.conv1d(
            outputs,
            conv_kernel,
            strides=self.strides[0],
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])
        #         print tf.shape(outputs)
        outputs = tf.reverse(outputs, axis=[1])
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        kernel_size = list(self.kernel_size)
        if self.kernel_size[0] % 2:
            kernel_size[0] = kernel_size[0]*2 -1
        else:
            kernel_size[0] = kernel_size[0]*2
        kernel_size = tuple(kernel_size)
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size_,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv1D_zerophase_linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Conv1D_linearphase(Layer):

    def __init__(self, filters, kernel_size, rank=1, strides=1, padding='valid', data_format='channels_last',
                 dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super(Conv1D_linearphase, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size_=kernel_size
        if kernel_size % 2:
            self.kernel_size = conv_utils.normalize_tuple(kernel_size // 2 + 1, rank, 'kernel_size')
        else:
            self.kernel_size = conv_utils.normalize_tuple(kernel_size // 2, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        if self.kernel_size[0] % 2 == 0:
            flipped = tf.reverse(self.kernel, axis=[0])
        else:
            flipped = tf.reverse(self.kernel[1:, :, :], axis=[0])
        #         print (flipped)
        conv_kernel = tf.concat([flipped, self.kernel], axis=0)
        #         print (conv_kernel)

        outputs = K.conv1d(
            inputs,
            conv_kernel,
            strides=self.strides[0],
            padding='same',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])
        #         print tf.shape(outputs)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        kernel_size = list(self.kernel_size)
        if self.kernel_size[0] % 2:
            kernel_size[0] = kernel_size[0]*2 -1
        else:
            kernel_size[0] = kernel_size[0]*2
        kernel_size = tuple(kernel_size)
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size_,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv1D_linearphase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Conv1D_linearphaseType(Layer):

    def __init__(self, filters, kernel_size, rank=1, strides=1, padding='valid', data_format='channels_last',
                 dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, type = 1, **kwargs):
        super(Conv1D_linearphaseType, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        # self.kernel_size_=kernel_size
        if type > 4:
            raise ValueError('FIR type should be between 1-4')
        else:
            self.type = type

        ## FIR type and Kernel Size Coherence Check
        if not type % 2 and kernel_size % 2:
            warnings.warn("Type %d FIR kernel size specified as %d. Using %d instead." % (type,kernel_size,kernel_size-1))
            kernel_size = kernel_size - 1
        elif type % 2 and not kernel_size % 2:
            warnings.warn(
            "Type %d FIR kernel size specified as %d. Using %d instead." % (type, kernel_size, kernel_size + 1))
            kernel_size = kernel_size + 1
        self.kernel_size_ = kernel_size
        if kernel_size % 2:
            self.kernel_size = conv_utils.normalize_tuple(kernel_size // 2 + 1, rank, 'kernel_size')
        else:
            self.kernel_size = conv_utils.normalize_tuple(kernel_size // 2, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        if self.kernel_size[0] % 2 == 0:
            flipped = tf.reverse(self.kernel, axis=[0])
        else:
            flipped = tf.reverse(self.kernel[1:, :, :], axis=[0])
        if self.type > 2:
            flipped = tf.multiply(-1., flipped)
        #         print (flipped)
        conv_kernel = tf.concat([flipped, self.kernel], axis=0)
        #         print (conv_kernel)

        outputs = K.conv1d(
            inputs,
            conv_kernel,
            strides=self.strides[0],
            padding='same',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])
        #         print tf.shape(outputs)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        kernel_size = list(self.kernel_size)
        if self.kernel_size[0] % 2:
            kernel_size[0] = kernel_size[0]*2 -1
        else:
            kernel_size[0] = kernel_size[0]*2
        kernel_size = tuple(kernel_size)
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size_,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv1D_linearphaseType, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Conv1D_linearphaseType_legacy(Layer):

    def __init__(self, filters, kernel_size, rank=1, strides=1, padding='valid', data_format='channels_last',
                 dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, type = 1, **kwargs):
        super(Conv1D_linearphaseType_legacy, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        # self.kernel_size_=kernel_size
        if type > 4:
            raise ValueError('FIR type should be between 1-4')
        else:
            self.type = type

        self.kernel_size_ = kernel_size
        if kernel_size % 2:
            self.kernel_size = conv_utils.normalize_tuple(kernel_size // 2 + 1, rank, 'kernel_size')
        else:
            self.kernel_size = conv_utils.normalize_tuple(kernel_size // 2, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        if self.kernel_size[0] % 2 == 0:
            flipped = tf.reverse(self.kernel, axis=[0])
        else:
            flipped = tf.reverse(self.kernel[1:, :, :], axis=[0])
        if self.type > 2:
            flipped = tf.multiply(-1., flipped)
        #         print (flipped)
        conv_kernel = tf.concat([flipped, self.kernel], axis=0)
        #         print (conv_kernel)

        outputs = K.conv1d(
            inputs,
            conv_kernel,
            strides=self.strides[0],
            padding='same',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])
        #         print tf.shape(outputs)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        kernel_size = list(self.kernel_size)
        if self.kernel_size[0] % 2:
            kernel_size[0] = kernel_size[0]*2 -1
        else:
            kernel_size[0] = kernel_size[0]*2
        kernel_size = tuple(kernel_size)
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size_,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv1D_linearphaseType_legacy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class DCT1D(Layer):

    def __init__(self, type=2, n=None, axis=-2, norm=None, rank=1, data_format='channels_last',**kwargs):
        super(DCT1D, self).__init__(**kwargs)
        self.rank = rank
        self.type = type
        self.n = n
        self.axis = axis
        self.norm = norm
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        if norm is not None:
            if norm != 'ortho':
                raise ValueError('Normalization should be `ortho` or `None`')


    def compute_output_shape(self, input_shape):
        if self.n is not None:
            space = list(input_shape)
            space[self.axis] = self.n
            return tuple(space)
        else:
            return input_shape

    def call(self, inputs):

        x = tf.transpose(inputs, [0, 2, 1])
        x = tf.spectral.dct(x, type=self.type, n=self.n, axis=-1, norm=self.norm)
        outputs = tf.transpose(x, [0, 2, 1])
        return outputs

    def get_config(self):
        config = {
            'rank': self.rank,
            'data_format': self.data_format,
            'type': self.type,
            'n': self.n,
            'axis': self.axis,
            'norm': self.norm,
        }
        base_config = super(DCT1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Conv1D_gammatone(Layer):

    def __init__(self, filters=1, kernel_size=80, rank=1, strides=1, padding='valid',
                 data_format='channels_last', dilation_rate=1, activation=None, use_bias=True,
                 fsHz=1000.,
                 fc_initializer=initializers.RandomUniform(minval=10, maxval=400),
                 n_order_initializer=initializers.constant(4.),
                 amp_initializer=initializers.constant(10 ** 5),
                 beta_initializer=initializers.RandomNormal(mean=30, stddev=6),
                 bias_initializer='zeros',
                 **kwargs):
        super(Conv1D_gammatone, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size_ = kernel_size
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.fc_initializer = initializers.get(fc_initializer)
        self.n_order_initializer = initializers.get(n_order_initializer)
        self.amp_initializer = initializers.get(amp_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.input_spec = InputSpec(ndim=self.rank + 2)

        self.fsHz = fsHz
        self.t = tf.range(start=0, limit=kernel_size / float(fsHz),
                          delta=1 / float(fsHz), dtype=K.floatx())
        self.t = tf.expand_dims(input=self.t, axis=-1)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)

        ## Add learnable parameters
        self.fc = self.add_weight(shape=(self.filters, 1),
                                  initializer=self.fc_initializer,
                                  name='fc')
        self.n_order = self.add_weight(shape=(1, 1),
                                       initializer=self.n_order_initializer,
                                       name='n')
        self.amp = self.add_weight(shape=(self.filters, 1),
                                   initializer=self.amp_initializer,
                                   name='a')
        self.beta = self.add_weight(shape=(self.filters, 1),
                                    initializer=self.beta_initializer,
                                    name='beta')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias')
        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        # Get gammatone impulse response

        gammatone = self.impulse_gammatone()
        gammatone = tf.expand_dims(gammatone, axis=-2)  ## Considering single input channel

        if self.kernel_shape[1] > 1:
            raise ValueError('Number of channels for input to gammatone layer'
                             'should be 1.')

        outputs = K.conv1d(
            inputs,
            gammatone,
            strides=self.strides[0],
            padding='same',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def impulse_gammatone(self):

        gammatone = tf.multiply(tf.multiply(
            tf.matmul(self.amp, tf.pow(x=self.t, y=self.n_order - 1),
                      transpose_b=True),
            tf.exp(tf.multiply(-2 * np.pi, tf.matmul(self.beta, self.t,
                                                     transpose_b=True)))),
            tf.cos(tf.multiply(2 * np.pi, tf.matmul(self.fc, self.t,
                                                    transpose_b=True))))
        gammatone = tf.transpose(gammatone)
        return gammatone

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size_,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'fsHz': self.fsHz,
            'fc_initializer': initializers.serialize(self.fc_initializer),
            'n_order_initializer': initializers.serialize(self.n_order_initializer),
            'amp_initializer': initializers.serialize(self.amp_initializer),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(Conv1D_gammatone, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))