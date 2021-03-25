from tensorflow.keras import backend
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from ..utils.base_model import BaseModel

backend.set_image_data_format('channels_first')


class ResNet(BaseModel):
    def __init__(self,
                 input_shape,
                 output_size=1,
                 name="ResNet50V2",
                 data_format="channels_first",
                 weights="imagenet",
                 dropout=0.0,
                 l2_lambda=0.0,
                 apply_custom_output=False,
                 final_activation="sigmoid",
                 *args,
                 **kwargs
                 ):
        self._input_shape = input_shape
        self._name = name
        self._output_size = output_size
        self._data_format = data_format
        self._weights = weights
        self._base = self._set_model_top()
        self._dropout = dropout
        self._l2_lambda = l2_lambda
        self._apply_custom_output = apply_custom_output
        self._final_activation = final_activation
        super(ResNet, self).__init__(*args, **kwargs)

    def model(self, *args, **kwargs):
        base_outputs = self._base.output
        x = self._custom_output(base_outputs)
        outputs = Dense(self._output_size, activation=self._final_activation,
                        name='resnet_output')(x)
        return Model(inputs=self._base.input, outputs=outputs, name='resnet_model')

    def _set_model_top(self):
        if self._name == "ResNet50V2":
            return ResNet50V2(
                include_top=False, weights=self._weights, input_shape=self._input_shape)
        else:
            raise NotImplementedError

    def _custom_output(self, x):
        if self._apply_custom_output:
            x = GlobalAveragePooling2D()(x)
            x = Dropout(self._dropout)(x)
            x = Dense(128, kernel_initializer='he_uniform', activity_regularizer=l2(self._l2_lambda))(x)
            return LeakyReLU()(x)
        else:
            return GlobalAveragePooling2D(data_format=self._data_format, name='avg_pool')(x)
