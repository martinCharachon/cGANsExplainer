from tensorflow.keras import backend
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from ..utils.base_model import BaseModel
from ..utils.base_blocks import EncoderBlock, UNetDecoderBlock, OutputblockGenerator
backend.set_image_data_format('channels_first')


class UNetGenerator(BaseModel):
    def __init__(self,
                 input_shape,
                 nb_filters: list = [16, 32, 64, 128],
                 downsampling_type: str = "MaxPooling2D",
                 nb_convolution_block_down=1,
                 nb_convolution_block_up=1,
                 use_context=False,
                 context_dropout=None,
                 upsampling_type: str = "UpSampling2D",
                 final_convolution_layer: str = "Conv2D",
                 final_convolution_block_filters: list = None,
                 kernel_size=3,
                 activation="relu",
                 normalization_type="batch_normalization",
                 kernel_regularizer=None,
                 data_format="channels_first",
                 convolution_dropout_down=None,
                 convolution_dropout_up=None,
                 dropout_type=None,
                 *args,
                 **kwargs
                 ):

        self._input_shape = input_shape
        self._encoder_block = EncoderBlock(
            nb_filters=nb_filters, downsampling_type=downsampling_type,
            nb_convolution_block_down=nb_convolution_block_down, use_context=use_context,
            context_dropout=context_dropout, kernel_size=kernel_size, activation=activation,
            normalization_type=normalization_type, kernel_regularizer=kernel_regularizer,
            data_format=data_format, convolution_dropout_down=convolution_dropout_down,
            dropout_type=dropout_type)

        self._decoder_block = UNetDecoderBlock(
            nb_filters=nb_filters, nb_convolution_block_up=nb_convolution_block_up,
            upsampling_type=upsampling_type, kernel_size=kernel_size, activation=activation,
            normalization_type=normalization_type, kernel_regularizer=kernel_regularizer,
            data_format=data_format, convolution_dropout_up=convolution_dropout_up,
            dropout_type=dropout_type)
        self._output_block = OutputblockGenerator(
            input_shape=self._input_shape, final_convolution_layer=final_convolution_layer,
            final_convolution_block_filters=final_convolution_block_filters,
            data_format=data_format)

        self.encoder_model = self._build_encoder_model()
        self.decoder_inputs_shape_list = self._encoder_block.ref_encoded_layer_shape_list
        self.decoder_model = self._build_decoder_model()
        super(UNetGenerator, self).__init__(*args, **kwargs)

    def model(self, *args, **kwargs):
        image_inputs = Input(shape=self._input_shape)
        encoded_outputs = self.encoder_model(image_inputs)
        decoded_outputs = self.decoder_model(encoded_outputs)
        return Model(inputs=image_inputs, outputs=[encoded_outputs, decoded_outputs],
                     name="interpreter")

    def _build_encoder_model(self):
        image_inputs = Input(shape=self._input_shape)
        outputs = self._encoder_block(image_inputs)
        return Model(inputs=image_inputs, outputs=outputs, name="encoder")

    def _build_decoder_model(self):
        inputs = [Input(shape=ref_layer_shape)
                  for ref_layer_shape in self.decoder_inputs_shape_list]
        decoded_output = self._decoder_block(inputs)
        outs = self._output_block(decoded_output)
        return Model(inputs=inputs, outputs=outs, name="decoder")




