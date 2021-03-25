from tensorflow.keras import backend
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from ..utils.base_model import BaseModel

backend.set_image_data_format('channels_first')


class VAEInceptionEncoder(BaseModel):
    def __init__(self,
                 input_shape,
                 name="InceptionV3",
                 output_size=256,
                 target_size=None,
                 data_format="channels_first",
                 weights="imagenet",
                 *args,
                 **kwargs
                 ):

        self._input_shape = input_shape
        self.target_size = target_size
        self._name = name
        self._output_size = output_size
        self._data_format = data_format
        self._weights = weights
        self._encoder = self._set_model_top()
        super(VAEInceptionEncoder, self).__init__(*args, **kwargs)

    def model(self, *args, **kwargs):
        base_outputs = self._encoder.output
        mu, logvar = self._output_function(base_outputs)
        mu = Dense(self._output_size, name='encoder_mu_output')(mu)
        logvar = Dense(self._output_size, name='encoder_logvar_output')(logvar)
        self.encoded_layer_shape = mu.shape[1:]
        return Model(inputs=self._encoder.input, outputs=[mu, logvar], name='encoder')

    def _set_model_top(self):
        if self._name == "InceptionV3":
            return InceptionV3(include_top=False, weights=self._weights,
                               input_shape=self._input_shape)
        else:
            raise NotImplementedError

    def _output_function(self, x):
        if self.target_size is None:
            self.target_size = x.shape[1:]
        logvar = GlobalAveragePooling2D(
            data_format=self._data_format, name='avg_pool_logvar')(x)
        mu = GlobalAveragePooling2D(data_format=self._data_format, name='avg_pool_mu')(x)
        return mu, logvar
