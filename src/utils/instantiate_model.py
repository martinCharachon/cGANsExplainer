from tensorflow.keras.models import load_model, model_from_json
from .base_layers import OutActivationLayer, ClipLayer

DEFAULT_CUSTOM_OBJECTS = {
    "OutActivationLayer": OutActivationLayer,
    "ClipLayer": ClipLayer
}


def instantiate_model(model_json_path=None,
                      weights_path=None,
                      model_path=None,
                      custom_objects=DEFAULT_CUSTOM_OBJECTS):
    if model_path is not None:
        model = load_model(model_path,
                           custom_objects=custom_objects)
    elif model_json_path is not None:
        json_file = open(model_json_path, 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        if weights_path is not None:
            model.load_weights(weights_path)
    else:
        model = None
    return model
