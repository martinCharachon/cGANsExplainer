import numpy as np
from keras_preprocessing import image
from tensorflow.keras import backend as K

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def set_geometric_preprocessor(config_json):
    if config_json is None:
        preprocessor = IdentityPreprocessor()
    elif config_json["name"] == "AugmentationPreprocessor2D":
        preprocessor = AugmentationPreprocessor(**config_json["parameters"])
    elif config_json["name"] == "BasicPreprocessor2D":
        preprocessor = BasicPreprocessor(**config_json["parameters"])
    else:
        preprocessor = IdentityPreprocessor()
    return preprocessor


class IdentityPreprocessor:
    def __init__(self, interpolation='nearest'):
        self.data_format = K.image_data_format()
        self.interpolation = interpolation

    def __call__(self, x: np.ndarray, target_size=None):
        return resize_image(
            x, target_size=target_size,
            data_format=self.data_format, interpolation=self.interpolation)


class BasicPreprocessor:

    def __init__(self, interpolation='nearest', rescaling_method="min_max"):
        self.data_format = K.image_data_format()
        self.interpolation = interpolation
        self.rescaling_method = rescaling_method

    def __call__(self, x: np.ndarray, target_size=None):
        x = resize_image(
            x, target_size=target_size,
            data_format=self.data_format, interpolation=self.interpolation)
        if isinstance(self.rescaling_method, list):
            for m in self.rescaling_method:
                x = rescale(x, m)
            return x
        return rescale(x, self.rescaling_method)


class AugmentationPreprocessor:

    def __init__(self,
                 rotation_range=0,
                 height_shift_range=0,
                 width_shift_range=0,
                 shear_range=0,
                 zoom_range=(1, 1),
                 horizontal_flip=False,
                 vertical_flip=False,
                 interpolation="nearest",
                 rescaling_method="min_max"):
        self.data_format = K.image_data_format()
        if self.data_format == "channels_first":
            self.row_axis = 2
            self.col_axis = 3
            self.channel_axis = 1
        elif self.data_format == "channels_last":
            raise NotImplementedError
        self.interpolation = interpolation
        self.rescaling_method = rescaling_method
        self.rotation_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def __call__(self, x: np.ndarray, target_size=None):
        x = resize_image(x, target_size=target_size,
                         data_format=self.data_format, interpolation=self.interpolation)
        x = self.random_transform(x)
        if isinstance(self.rescaling_method, list):
            for m in self.rescaling_method:
                x = rescale(x, m)
            return x
        return rescale(x, self.rescaling_method)

    def get_random_transform(self, img_shape, seed=None):
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1

        if seed is not None:
            np.random.seed(seed)

        if self.rotation_range:
            theta = np.random.uniform(
                -self.rotation_range,
                self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            try:  # 1-D array-like or int
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tx = np.random.uniform(-self.height_shift_range,
                                       self.height_shift_range)
            if np.max(self.height_shift_range) < 1:
                tx *= img_shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            try:  # 1-D array-like or int
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(-self.width_shift_range,
                                       self.width_shift_range)
            if np.max(self.width_shift_range) < 1:
                ty *= img_shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(
                -self.shear_range,
                self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0],
                self.zoom_range[1],
                2)
        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        transform_parameters = {'theta': theta,
                                'tx': tx,
                                'ty': ty,
                                'shear': shear,
                                'zx': zx,
                                'zy': zy,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical}

        return transform_parameters

    def apply_transform(self, x, transform_parameters):
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        x = image.apply_affine_transform(
            x,
            transform_parameters.get('theta', 0),
            transform_parameters.get('tx', 0),
            transform_parameters.get('ty', 0),
            transform_parameters.get('shear', 0),
            transform_parameters.get('zx', 1),
            transform_parameters.get('zy', 1),
            row_axis=img_row_axis,
            col_axis=img_col_axis,
            channel_axis=img_channel_axis,
            fill_mode='nearest',
            cval=0.,
            order=1)

        if transform_parameters.get('flip_horizontal', False):
            x = image.flip_axis(x, img_col_axis)

        if transform_parameters.get('flip_vertical', False):
            x = image.flip_axis(x, img_row_axis)

        return x

    def apply_transform_inverse(self, x, transform_parameters):
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        transform_params_inverse = \
            {'theta': - transform_parameters["theta"],
             'tx': - transform_parameters["tx"],
             'ty': - transform_parameters["ty"],
             'shear': 0,
             'zx': 1 / transform_parameters["zx"],
             'zy': 1 / transform_parameters["zy"],
             'flip_horizontal': transform_parameters["flip_horizontal"],
             'flip_vertical': transform_parameters["flip_vertical"]}

        if transform_params_inverse.get('flip_vertical', False):
            x = image.flip_axis(x, img_row_axis)

        if transform_params_inverse.get('flip_horizontal', False):
            x = image.flip_axis(x, img_col_axis)

        x = image.apply_affine_transform(
            x,
            transform_params_inverse.get('theta', 0),
            transform_params_inverse.get('tx', 0),
            transform_params_inverse.get('ty', 0),
            transform_params_inverse.get('shear', 0),
            transform_params_inverse.get('zx', 1),
            transform_params_inverse.get('zy', 1),
            row_axis=img_row_axis,
            col_axis=img_col_axis,
            channel_axis=img_channel_axis,
            fill_mode='nearest',
            cval=0.,
            order=1)

        return x

    def random_transform(self, x, seed=None):
        params = self.get_random_transform(x.shape, seed)
        return self.apply_transform(x, params)


def resize_image(x, target_size=None, interpolation='nearest', data_format="channels_first"):
    img = image.utils.array_to_img(x, data_format=data_format)
    if target_size is not None:
        width_height_tuple = (target_size[2], target_size[1])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return image.utils.img_to_array(img, data_format=data_format)


def rescale(x, method="min_max"):
    if method == "min_max":
        eps = 0
        if (np.max(x) - np.min(x)) < 1e-8:
            eps = 1e-8
        return (x - np.min(x)) / (np.max(x) - np.min(x) + eps)
    elif method == "mean_std":
        eps = 0
        if np.std(x) < 1e-8:
            eps = 1e-8
        return (x - np.mean(x)) / (np.std(x) + eps)
    elif method is None:
        return x
    else:
        raise NotImplementedError
