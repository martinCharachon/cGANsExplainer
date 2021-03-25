#!/usr/bin/env python
import importlib
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, \
    Activation, Conv2DTranspose, SpatialDropout2D, Add, \
    MaxPooling2D, AveragePooling2D, UpSampling2D, Concatenate, Layer


def downsampling_block(in_layer,
                       nb_filters,
                       dwn_type="Conv2D",
                       dwn_convolution=True,
                       pool_size=(2, 2),
                       normalization_type="batch_normalization",
                       activation="relu",
                       kernel_size=(3, 3),
                       kernel_regularizer=None,
                       kernel_initializer="glorot_uniform",
                       convolution_dropout=None,
                       dropout_type=None,
                       name=None):

    if dwn_type == "MaxPooling2D":
        if dwn_convolution:
            in_layer = convolution_block(layer=in_layer,
                                         nb_filters=nb_filters,
                                         normalization_type=normalization_type,
                                         activation=activation,
                                         kernel_size=kernel_size,
                                         kernel_regularizer=kernel_regularizer,
                                         kernel_initializer=kernel_initializer,
                                         convolution_dropout=convolution_dropout,
                                         dropout_type=dropout_type)
        in_layer = MaxPooling2D(pool_size=pool_size,
                                strides=None,
                                data_format="channels_first",
                                padding="same",
                                name=name)(in_layer)

    elif dwn_type == "AveragePooling2D":
        if dwn_convolution:
            in_layer = convolution_block(layer=in_layer,
                                         nb_filters=nb_filters,
                                         normalization_type=normalization_type,
                                         activation=activation,
                                         kernel_size=kernel_size,
                                         kernel_regularizer=kernel_regularizer,
                                         kernel_initializer=kernel_initializer,
                                         convolution_dropout=convolution_dropout,
                                         dropout_type=dropout_type)

        in_layer = AveragePooling2D(pool_size=pool_size,
                                    strides=None,
                                    padding="same",
                                    data_format='channels_first',
                                    name=name)(in_layer)

    elif dwn_type == "Conv2D":
        in_layer = convolution_block(layer=in_layer,
                                     nb_filters=nb_filters,
                                     strides=pool_size,
                                     normalization_type=normalization_type,
                                     activation=activation,
                                     kernel_size=kernel_size,
                                     kernel_regularizer=kernel_regularizer,
                                     kernel_initializer=kernel_initializer,
                                     convolution_dropout=convolution_dropout,
                                     dropout_type=dropout_type)
    else:
        in_layer = convolution_block(layer=in_layer,
                                     nb_filters=nb_filters,
                                     strides=pool_size,
                                     normalization_type=normalization_type,
                                     activation=activation,
                                     kernel_size=kernel_size,
                                     kernel_regularizer=kernel_regularizer,
                                     kernel_initializer=kernel_initializer,
                                     convolution_dropout=convolution_dropout,
                                     dropout_type=dropout_type)

    return in_layer


def upsampling_module(in_layer,
                      nb_filters,
                      size,
                      nb_convolution_block=1,
                      type="UpSampling2D",
                      data_format='channels_first',
                      **kwargs):
    if type == "Conv2DTranspose":
        in_layer = convolution_block_layers(
            layer=in_layer, nb_convolution_block=nb_convolution_block,
            nb_filters=nb_filters, data_format=data_format, **kwargs)
        in_layer = Conv2DTranspose(nb_filters,
                                   kernel_size=(3, 3),
                                   strides=size,
                                   padding='same',
                                   data_format=data_format)(in_layer)
    elif type == "UpSampling2D":
        in_layer = convolution_block_layers(
            layer=in_layer, nb_convolution_block=nb_convolution_block,
            nb_filters=nb_filters, data_format=data_format, **kwargs)
        in_layer = UpSampling2D(size=size,
                                data_format=data_format)(in_layer)

    elif type == "UpSampling2D + Conv2DTranspose":
        in_layer = convolution_transpose_block_layers(
            layer=in_layer, nb_convolution_block=nb_convolution_block,
            nb_filters=nb_filters, data_format=data_format, **kwargs)

        in_layer = UpSampling2D(size=size,
                                data_format=data_format)(in_layer)

    elif type == "Conv2DTranspose + Conv2DTranspose":
        in_layer = convolution_transpose_block_layers(
            layer=in_layer, nb_convolution_block=nb_convolution_block,
            nb_filters=nb_filters, data_format=data_format, **kwargs)
        in_layer = Conv2DTranspose(nb_filters,
                                   kernel_size=(3, 3),
                                   strides=size,
                                   padding='same',
                                   data_format=data_format)(in_layer)
    else:
        raise NotImplementedError

    return in_layer


def upsampling_block(layer,
                     skip_connection_layer,
                     nb_filters,
                     nb_convolution_block=1,
                     size=(2, 2),
                     type="UpSampling2D",
                    data_format='channels_first',
                     **kwargs):
    layer = upsampling_module(in_layer=layer,
                              nb_filters=nb_filters,
                              nb_convolution_block=nb_convolution_block,
                              size=size,
                              type=type,
                              data_format=data_format,
                              **kwargs)
    layer = Concatenate(axis=1)([skip_connection_layer, layer])

    return layer


def residual_block(in_layer,
                   nb_filters,
                   nb_convolution_block=1,
                   dropout_rate=None,
                   use_context=False,
                   data_format='channels_first',
                   **kwargs):
    layer = convolution_block_layers(
        layer=in_layer,
        nb_filters=nb_filters,
        nb_convolution_block=nb_convolution_block,
        data_format=data_format,
        **kwargs)
    if dropout_rate is not None:
        layer = SpatialDropout2D(rate=dropout_rate, data_format=data_format)(layer)
    layer = convolution_block(layer=layer, nb_filters=nb_filters, data_format=data_format,
                                 **kwargs)
    if use_context:
        layer = Add()([in_layer, layer])

    return layer


def convolution_block(
        layer,
        nb_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        normalization_type="batch_normalization",
        padding='same',
        activation=None,
        kernel_initializer="glorot_uniform",
        use_bias=True,
        kernel_regularizer=None,
        dilation_rate=(1, 1),
        data_format='channels_first',
        convolution_dropout=None,
        dropout_type=None,
        activation_post_conv=False):

    with tf.name_scope('convolution_block'):
        layer = normalization_layer(layer,
                                    normalization_type)
        if not activation_post_conv:
            layer = activation_layer(layer,
                                     activation)
        layer = Conv2D(filters=nb_filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding=padding,
                       kernel_initializer=kernel_initializer,
                       use_bias=use_bias,
                       dilation_rate=dilation_rate,
                       kernel_regularizer=kernel_regularizer,
                       data_format=data_format)(layer)
        if activation_post_conv:
            layer = activation_layer(layer,
                                     activation)
        layer = dropout_layer(layer,
                              dropout=convolution_dropout,
                              dropout_type=dropout_type)

        return layer


def convolution_block_transpose(
        layer,
        nb_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        normalization_type="batch_normalization",
        padding='same',
        activation=None,
        kernel_initializer="glorot_uniform",
        use_bias=True,
        kernel_regularizer=None,
        dilation_rate=(1, 1),
        data_format='channels_first',
        convolution_dropout=None,
        dropout_type=None,
        name=None):

    layer = normalization_layer(layer,
                                normalization_type)
    layer = activation_layer(layer,
                             activation)
    layer = Conv2DTranspose(filters=nb_filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            kernel_initializer=kernel_initializer,
                            use_bias=use_bias,
                            dilation_rate=dilation_rate,
                            kernel_regularizer=kernel_regularizer,
                            data_format=data_format,
                            name=name)(layer)
    layer = dropout_layer(layer,
                          dropout=convolution_dropout,
                          dropout_type=dropout_type)

    return layer


def convolution_block_layers(layer,
                             nb_filters,
                             nb_convolution_block=None,
                             **kwargs):
    if not nb_convolution_block:
        return layer
    for num_conv in range(nb_convolution_block):
        layer = convolution_block(layer=layer,
                                     nb_filters=nb_filters,
                                     **kwargs)
    return layer


def convolution_transpose_block_layers(layer,
                                       nb_filters,
                                       nb_convolution_block=None,
                                       **kwargs):
    if not nb_convolution_block:
        return layer
    for num_conv in range(nb_convolution_block):
        layer = convolution_block_transpose(
            layer=layer,
            nb_filters=nb_filters,
            **kwargs)
    return layer


def normalization_layer(in_layer, kind=None):
    if kind == "batch_normalization":
        return BatchNormalization(axis=1)(in_layer)
    elif kind == "layer_normalization":
        return tf.keras.layers.LayerNormalization(axis=1)(in_layer)
    elif kind is None:
        return in_layer
    else:
        raise NotImplementedError


def activation_layer(in_layer, acti=None):
    if acti is None:
        return in_layer
    elif acti == "LeakyReLU":
        return LeakyReLU()(in_layer)
    else:
        return Activation(acti)(in_layer)


def dropout_layer(in_layer, dropout, dropout_type):
    if dropout is None:
        return in_layer
    else:
        if dropout_type is None:
            dropout_type = "Dropout"
        module = importlib.import_module("tensorflow.keras.layers")
        assigned_dropout_layer = getattr(module, dropout_type)
        return assigned_dropout_layer(rate=dropout)(in_layer)


def set_nb_convolution_block(nb_convolution_block, network_depth):
    if not isinstance(nb_convolution_block, list):
        return [nb_convolution_block] * network_depth
    else:
        return nb_convolution_block


class OutActivationLayer(Layer):
    def __init__(self, activation=None, **kwargs):
        super(OutActivationLayer, self).__init__(**kwargs)
        self.activation = activation
        if self.activation is not None:
            self.activation_function = getattr(tf.keras.activations, self.activation)

    def call(self, inputs, **kwargs):
        if self.activation is not None:
            tensor = tf.reduce_mean(inputs, axis=1, keepdims=True)
            return self.activation_function(tensor)

        a = tf.expand_dims(tf.abs(inputs[:, 0, ...]), 1)
        b = tf.expand_dims(tf.abs(inputs[:, 1, ...]), 1)
        return a / (a + b)

    def get_config(self):
        config = {
            'activation': self.activation
        }
        base_config = super(OutActivationLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class ClipLayer(Layer):
    def __init__(self, value_min=0, value_max=1, **kwargs):
        super(ClipLayer, self).__init__(**kwargs)
        self.value_min = value_min
        self.value_max = value_max

    def call(self, inputs, **kwargs):
        clipped_inputs = tf.keras.backend.clip(inputs, self.value_min, self.value_max)
        return clipped_inputs

    def get_config(self):
        config = {
            'value_min': self.value_min,
            'value_max': self.value_max,
        }
        base_config = super(ClipLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape



