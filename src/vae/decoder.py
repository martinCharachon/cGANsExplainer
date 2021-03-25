from tensorflow.keras import backend
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from ..utils.base_model import BaseModel
from ..utils.base_blocks import DecoderBlock, OutputblockGenerator

backend.set_image_data_format('channels_first')


class Decoder(BaseModel):
    def __init__(self,
                 input_shape,
                 encoding_dimension: list,
                 target_size: list,
                 nb_filters: list = [16, 32, 64, 128],
                 nb_convolution_block_up=1,
                 upsampling_type: str = "Conv2DTranspose",
                 final_convolution_layer: str = "Conv2DTranspose",
                 final_convolution_block_filters: list = None,
                 kernel_size=3,
                 activation="relu",
                 normalization_type="batch_normalization",
                 kernel_regularizer=None,
                 data_format="channels_first",
                 convolution_dropout_up=None,
                 dropout_type=None,
                 *args,
                 **kwargs
                 ):
        self._input_shape = input_shape
        self._encoding_dimension = encoding_dimension
        self._target_size = target_size
        self._decoder_block = DecoderBlock(
            target_size=self._target_size,
            nb_filters=nb_filters, nb_convolution_block_up=nb_convolution_block_up,
            upsampling_type=upsampling_type, kernel_size=kernel_size, activation=activation,
            normalization_type=normalization_type, kernel_regularizer=kernel_regularizer,
            data_format=data_format, convolution_dropout_up=convolution_dropout_up,
            dropout_type=dropout_type)
        self._output_block = OutputblockGenerator(
            input_shape=self._input_shape, final_convolution_layer=final_convolution_layer,
            final_convolution_block_filters=final_convolution_block_filters,
            data_format=data_format)
        super(Decoder, self).__init__(*args, **kwargs)

    def model(self, *args, **kwargs):
        inputs = Input(shape=self._encoding_dimension)
        decoded_outputs = self._decoder_block(inputs)
        outs = self._output_block(decoded_outputs)
        return Model(inputs=inputs, outputs=outs, name="decoder")
