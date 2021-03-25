from tensorflow.keras import backend
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from ..utils.base_model import BaseModel
from ..utils.base_blocks import EncoderBlock, OutputblockEncoder
backend.set_image_data_format('channels_first')


class VAEEncoder(BaseModel):
    def __init__(self,
                 input_shape,
                 encoded_dimension=256,
                 encoding_method="GlobalAveragePooling2D",
                 nb_filters: list = [16, 32, 64, 128],
                 downsampling_type: str = "Conv2D",
                 nb_convolution_block_down=1,
                 use_context=False,
                 context_dropout=None,
                 encoder_output_activation=None,
                 kernel_size=3,
                 activation="relu",
                 normalization_type="batch_normalization",
                 kernel_regularizer=None,
                 data_format="channels_first",
                 convolution_dropout_down=None,
                 dropout_type=None,
                 *args,
                 **kwargs
                 ):

        self._input_shape = input_shape
        self._encoded_dimension = encoded_dimension
        self._input_shape = input_shape
        self._encoder_block = EncoderBlock(
            nb_filters=nb_filters, downsampling_type=downsampling_type,
            nb_convolution_block_down=nb_convolution_block_down, use_context=use_context,
            context_dropout=context_dropout, kernel_size=kernel_size, activation=activation,
            normalization_type=normalization_type, kernel_regularizer=kernel_regularizer,
            data_format=data_format, convolution_dropout_down=convolution_dropout_down,
            dropout_type=dropout_type)
        self._outblock_mu = OutputblockEncoder(
            outputs_dimension=encoded_dimension,
            final_encoding_layer=encoding_method,
            final_activation=encoder_output_activation,
            kernel_size=kernel_size,
            data_format=data_format)
        self._outblock_logvar = OutputblockEncoder(
            outputs_dimension=encoded_dimension,
            final_encoding_layer=encoding_method,
            final_activation=encoder_output_activation,
            kernel_size=kernel_size,
            data_format=data_format)
        super(VAEEncoder, self).__init__(*args, **kwargs)

    def model(self, *args, **kwargs):
        image_inputs = Input(shape=self._input_shape)
        encoded_outputs = self._encoder_block(image_inputs)
        mu = self._outblock_mu(encoded_outputs[-1])
        logvar = self._outblock_logvar(encoded_outputs[-1])
        self.target_size = self._encoder_block.target_size
        self.encoded_layer_shape = mu.shape[1:]
        return Model(inputs=image_inputs, outputs=[mu, logvar], name="encoder")


