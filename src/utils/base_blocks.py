#!/usr/bin/env python
from tensorflow.keras.layers import Conv2D, Activation, Conv2DTranspose, Concatenate, \
    Flatten, Dense, GlobalAveragePooling2D, Reshape, SpatialDropout2D
from .base_layers import convolution_block_layers, downsampling_block, \
    residual_block, upsampling_block, upsampling_module, set_nb_convolution_block, convolution_block


class EncoderBlock:

    def __init__(self,
                 nb_filters: list = [16, 32, 64, 128],
                 downsampling_type: str = "MaxPooling2D",
                 nb_convolution_block_down=1,
                 use_context=False,
                 context_dropout=None,
                 kernel_size=3,
                 activation="relu",
                 normalization_type="batch_normalization",
                 kernel_regularizer=None,
                 data_format="channels_first",
                 convolution_dropout_down=None,
                 dropout_type=None):
        self._nb_filters = nb_filters
        self._network_depth = len(self._nb_filters)
        self._nb_convolution_block_down = set_nb_convolution_block(
            nb_convolution_block_down, self._network_depth)
        assert len(self._nb_convolution_block_down) == self._network_depth
        self._downsampling_type = downsampling_type
        self._use_context = use_context
        self._context_dropout = context_dropout
        self._activation = activation
        self._normalization_type = normalization_type
        self._kernel_size = kernel_size
        self._kernel_regularizer = kernel_regularizer
        self._data_format = data_format
        self._convolution_dropout_down = convolution_dropout_down
        self._dropout_type = dropout_type

    def __call__(self, inputs):
        encoded_layers = []
        layer = convolution_block_layers(
            layer=inputs,
            nb_filters=self._nb_filters[0],
            nb_convolution_block=self._nb_convolution_block_down[0],
            kernel_size=self._kernel_size,
            activation=self._activation,
            normalization_type=self._normalization_type,
            kernel_regularizer=self._kernel_regularizer,
            convolution_dropout=self._convolution_dropout_down,
            dropout_type=self._dropout_type
        )
        encoded_layers.append(layer)

        for level_number in range(1, self._network_depth):
            layer = downsampling_block(in_layer=layer,
                                       nb_filters=self._nb_filters[level_number],
                                       dwn_type=self._downsampling_type,
                                       normalization_type=self._normalization_type,
                                       kernel_size=self._kernel_size,
                                       activation=self._activation,
                                       kernel_regularizer=self._kernel_regularizer,
                                       convolution_dropout=self._convolution_dropout_down,
                                       dropout_type=self._dropout_type
                                       )
            layer = residual_block(
                layer,
                nb_filters=self._nb_filters[level_number],
                nb_convolution_block=self._nb_convolution_block_down[level_number],
                use_context=self._use_context,
                dropout_rate=self._context_dropout,
                normalization_type=self._normalization_type,
                kernel_size=self._kernel_size,
                activation=self._activation,
                kernel_regularizer=self._kernel_regularizer,
                data_format=self._data_format,
                convolution_dropout=self._convolution_dropout_down,
                dropout_type=self._dropout_type
            )

            encoded_layers.append(layer)
        self.target_size = layer.shape[1:]
        self.ref_encoded_layer_shape_list = [layer.shape[1:] for layer in encoded_layers]
        return encoded_layers


class FCNBlock:

    def __init__(self,
                 nb_filters: list = [16, 32, 64, 128],
                 downsampling_type: str = "MaxPooling2D",
                 nb_convolution_block=1,
                 kernel_size=3,
                 activation="relu",
                 normalization_type="batch_normalization",
                 kernel_regularizer=None,
                 data_format="channels_first",
                 spatial_dropout=None,
                 convolution_dropout=None,
                 dropout_type=None):
        self._nb_filters = nb_filters
        self._network_depth = len(self._nb_filters)
        self._nb_convolution_block = set_nb_convolution_block(
            nb_convolution_block, self._network_depth)
        assert len(self._nb_convolution_block) == self._network_depth
        self._downsampling_type = downsampling_type
        self._activation = activation
        self._normalization_type = normalization_type
        self._kernel_size = kernel_size
        self._kernel_regularizer = kernel_regularizer
        self._data_format = data_format
        self._spatial_dropout = spatial_dropout
        self._convolution_dropout = convolution_dropout
        self._dropout_type = dropout_type

    def __call__(self, inputs):
        layer = inputs
        for i in range(self._network_depth):
            layer = convolution_block_layers(
                layer=layer,
                nb_filters=self._nb_filters[i],
                nb_convolution_block=self._nb_convolution_block[i],
                activation=self._activation,
                normalization_type=self._normalization_type,
                kernel_regularizer=self._kernel_regularizer,
                convolution_dropout=self._convolution_dropout,
                dropout_type=self._dropout_type)
            if self._spatial_dropout is not None:
                layer = SpatialDropout2D(rate=self._spatial_dropout)(layer)
            layer = downsampling_block(layer,
                                       nb_filters=self._nb_filters[i],
                                       dwn_type=self._downsampling_type,
                                       normalization_type=self._normalization_type,
                                       activation=self._activation,
                                       convolution_dropout=self._convolution_dropout,
                                       dropout_type=self._dropout_type)
        return layer


class UNetDecoderBlock:
    def __init__(self,
                 nb_filters: list = [16, 32, 64, 128],
                 nb_convolution_block_up=1,
                 upsampling_type: str = "UpSampling2D",
                 kernel_size=3,
                 activation="relu",
                 normalization_type="batch_normalization",
                 kernel_regularizer=None,
                 data_format="channels_first",
                 convolution_dropout_up=None,
                 dropout_type=None
                 ):

        self._nb_filters = nb_filters
        self._network_depth = len(self._nb_filters)

        self._nb_convolution_block_up = set_nb_convolution_block(
            nb_convolution_block_up, self._network_depth - 1)
        assert len(self._nb_convolution_block_up) == self._network_depth - 1
        self._up_type = upsampling_type
        self._activation = activation
        self._normalization_type = normalization_type
        self._kernel_size = kernel_size
        self._kernel_regularizer = kernel_regularizer
        self._data_format = data_format
        self._convolution_dropout_up = convolution_dropout_up
        self._dropout_type = dropout_type

    def __call__(self, inputs):
        skip_connection_layers, layer = inputs[:-1], inputs[-1]
        for level_number in range(self._network_depth - 2, -1, -1):
            skip_connection_layer = skip_connection_layers[level_number]
            layer = upsampling_block(
                layer,
                skip_connection_layer,
                nb_filters=self._nb_filters[level_number],
                nb_convolution_block=self._nb_convolution_block_up[level_number],
                size=(2, 2),
                type=self._up_type,
                normalization_type=self._normalization_type,
                activation=self._activation,
                kernel_size=self._kernel_size,
                kernel_regularizer=self._kernel_regularizer,
                convolution_dropout=self._convolution_dropout_up,
                dropout_type=self._dropout_type
            )

        return layer


class DecoderBlock:
    def __init__(self,
                 target_size,
                 nb_filters: list = [16, 32, 64, 128],
                 nb_convolution_block_up=1,
                 upsampling_type: str = "UpSampling2D",
                 kernel_size=3,
                 activation="relu",
                 normalization_type="batch_normalization",
                 kernel_regularizer=None,
                 data_format="channels_first",
                 convolution_dropout_up=None,
                 dropout_type=None
                 ):
        self._target_size = target_size
        self._nb_filters = nb_filters
        self._network_depth = len(self._nb_filters)

        self._nb_convolution_block_up = set_nb_convolution_block(
            nb_convolution_block_up, self._network_depth - 1)
        assert len(self._nb_convolution_block_up) == self._network_depth - 1
        self._up_type = upsampling_type
        self._activation = activation
        self._normalization_type = normalization_type
        self._kernel_size = kernel_size
        self._kernel_regularizer = kernel_regularizer
        self._data_format = data_format
        self._convolution_dropout_up = convolution_dropout_up
        self._dropout_type = dropout_type

    def __call__(self, inputs):
        layer = Dense(1 * self._target_size[1] * self._target_size[2], activation="relu")(inputs)
        layer = Reshape(target_shape=(1, self._target_size[1], self._target_size[2]))(layer)
        layer = convolution_block(
            layer,
            nb_filters=self._target_size[0],
            kernel_size=self._kernel_size,
            kernel_regularizer=self._kernel_regularizer,
            activation=self._activation,
            normalization_type=self._normalization_type)
        for level_number in range(self._network_depth - 2, -1, -1):
            layer = upsampling_module(
                layer,
                nb_filters=self._nb_filters[level_number],
                nb_convolution_block=self._nb_convolution_block_up[level_number],
                size=(2, 2),
                type=self._up_type,
                normalization_type=self._normalization_type,
                activation=self._activation,
                kernel_size=self._kernel_size,
                kernel_regularizer=self._kernel_regularizer,
                convolution_dropout=self._convolution_dropout_up,
                dropout_type=self._dropout_type
            )

        return layer


class OutputblockGenerator:
    def __init__(self,
                 input_shape,
                 final_convolution_layer: str = "Conv2D",
                 final_convolution_block_filters: list = None,
                 data_format="channels_first"
                 ):

        self._input_shape = input_shape
        self._final_convolution_layer = final_convolution_layer
        self._final_convolution_block_filters = final_convolution_block_filters
        self._data_format = data_format

    def __call__(self, layer):
        if self._final_convolution_block_filters is not None:
            for f in self._final_convolution_block_filters:
                layer = Conv2D(
                    f, kernel_size=3, padding="same", data_format=self._data_format)(layer)
        if self._final_convolution_layer == "Conv2D":
            out_layer = Conv2D(1, kernel_size=1, padding="same", data_format=self._data_format)(
                layer)
        elif self._final_convolution_layer == "Conv2DTranspose":
            out_layer = Conv2DTranspose(1, kernel_size=1, padding="same",
                                        data_format=self._data_format)(layer)
        else:
            raise KeyError
        if self._input_shape[0] == 3:
            out_layer = Concatenate(axis=1)([out_layer, out_layer, out_layer])
        return out_layer


class OutputblockEncoder:
    def __init__(self,
                 outputs_dimension: int = 1,
                 final_encoding_layer: str = "Dense",
                 final_activation: str = None,
                 kernel_size=3,
                 data_format="channels_first"
                 ):

        self._outputs_dimension = outputs_dimension
        self._final_encoding_layer = final_encoding_layer
        self._final_activation = final_activation
        self._kernel_size = kernel_size
        self._data_format = data_format

    def __call__(self, layer):
        if self._final_encoding_layer == "Conv2D + Dense":
            layer = Conv2D(1, kernel_size=self._kernel_size, padding="same",
                           data_format=self._data_format)(layer)
            layer = Flatten(data_format=self._data_format)(layer)
            output_layer = Dense(self._outputs_dimension, activation=self._final_activation)(layer)

        elif self._final_encoding_layer == "Conv2D":
            layer = Conv2D(1, kernel_size=self._kernel_size, padding="same",
                           data_format=self._data_format)(layer)
            output_layer = Activation(self._final_activation)(layer)

        elif self._final_encoding_layer == "Dense":
            layer = Flatten(data_format=self._data_format)(layer)
            output_layer = Dense(self._outputs_dimension, activation=self._final_activation)(layer)

        elif self._final_encoding_layer == "GlobalAveragePooling2D + Dense":
            layer = GlobalAveragePooling2D(data_format='channels_first')(layer)
            output_layer = Dense(self._outputs_dimension, activation=self._final_activation)(layer)

        elif self._final_encoding_layer == "GlobalAveragePooling2D":
            layer = Conv2D(self._outputs_dimension, kernel_size=self._kernel_size, padding="same",
                           data_format=self._data_format)(layer)
            layer = Conv2D(self._outputs_dimension, kernel_size=self._kernel_size, padding="same",
                           data_format=self._data_format)(layer)
            layer = GlobalAveragePooling2D(data_format='channels_first')(layer)
            output_layer = Activation(self._final_activation)(layer)
        else:
            raise AttributeError
        return output_layer


