from tensorflow.keras.layers import Conv2D, Input, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from ..utils.base_model import BaseModel


class LeNet(BaseModel):
    def __init__(self,
                 input_shape,
                 outputs_dimension=1,
                 nb_filters: list = [32, 64],
                 final_activation="sigmoid",
                 kernel_size=3,
                 activation="relu",
                 kernel_regularizer=None,
                 data_format="channels_first",
                 *args,
                 **kwargs
                 ):
        self._outputs_dimension = outputs_dimension
        self._input_shape = input_shape
        self._nb_filters = nb_filters
        self._final_activation = final_activation
        self._activation = activation
        self._kernel_size = kernel_size
        self._kernel_regularizer = kernel_regularizer
        self._data_format = data_format
        super(LeNet, self).__init__(*args, **kwargs)

    def model(self, *args, **kwargs):
        image_inputs = Input(shape=self._input_shape)
        outputs = self._encoder_block(image_inputs)
        return Model(inputs=image_inputs, outputs=outputs, name="lenet_model")

    def _encoder_block(self, image_layer):
        layer = Conv2D(
            self._nb_filters[0], kernel_size=self._kernel_size,
            activation=self._activation, data_format=self._data_format)(
            image_layer)
        layer = Conv2D(
            self._nb_filters[1], kernel_size=self._kernel_size,
            activation=self._activation, data_format=self._data_format)(layer)
        layer = MaxPooling2D(pool_size=(2, 2), data_format=self._data_format)(layer)
        layer = Dropout(0.25)(layer)
        layer = Flatten()(layer)
        layer = Dense(128, activation=self._activation)(layer)
        layer = Dropout(0.5)(layer)
        return Dense(self._outputs_dimension, activation=self._final_activation)(layer)
