from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from ..utils.base_model import BaseModel
from ..utils.base_blocks import OutputblockEncoder, FCNBlock


class Discriminator(BaseModel):
    def __init__(self,
                 input_shape,
                 outputs_dimension=1,
                 nb_filters: list = [16, 32, 64, 92],
                 nb_convolution_block=1,
                 downsampling_type: str = "Conv2D",
                 final_encoding_layer="Dense",
                 final_activation=None,
                 kernel_size=3,
                 activation="LeakyReLU",
                 normalization_type=None,
                 kernel_regularizer=None,
                 data_format="channels_first",
                 *args,
                 **kwargs
                 ):

        self._final_encoding_layer = final_encoding_layer
        self._outputs_dimension = outputs_dimension
        self._input_shape = input_shape
        self._nb_filters = nb_filters
        self._nb_convolution_block = nb_convolution_block
        self._downsampling_type = downsampling_type
        self._final_activation = final_activation
        self._activation = activation
        self._normalization_type = normalization_type
        self._kernel_size = kernel_size
        self._kernel_regularizer = kernel_regularizer
        self._data_format = data_format
        self._fcn_block = FCNBlock(
            nb_filters=nb_filters, downsampling_type=downsampling_type,
            nb_convolution_block=nb_convolution_block, kernel_size=kernel_size,
            activation=activation, normalization_type=normalization_type,
            kernel_regularizer=kernel_regularizer, data_format=data_format)
        self._outblock = OutputblockEncoder(
            outputs_dimension=outputs_dimension,
            final_encoding_layer=final_encoding_layer,
            final_activation=final_activation,
            kernel_size=kernel_size,
            data_format=data_format)
        super(Discriminator, self).__init__(*args, **kwargs)

    def model(self, *args, **kwargs):
        image_inputs = Input(shape=self._input_shape)
        encoder_outputs = self._fcn_block(image_inputs)
        outputs = self._outblock(encoder_outputs)
        return Model(inputs=image_inputs, outputs=outputs, name="discriminator")

