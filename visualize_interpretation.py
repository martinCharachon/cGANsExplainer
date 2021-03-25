import json
import numpy as np
import configargparse
import h5py
import tensorflow as tf

from src.utils.visualization_tools import show_multi_visualization
from src.utils.evaluation_functions import load_json_param
from src.utils.instantiate_model import instantiate_model
from src.utils.custom_outputs_activation import CustomOutputActivation


def main():
    p = configargparse.ArgParser()
    p.add_argument('--config-file', required=True)
    p.add_argument('--ref-model-settings-json-path', required=False)
    p.add_argument('--indexes-name', required=False, default="test_indexes")
    p.add_argument('--threshold', required=False, default=95, type=int)
    args = p.parse_args()
    config = load_json_param(args.config_file)
    json_file = open(args.ref_model_settings_json_path, 'r', encoding="utf8", errors='ignore')
    settings = json.load(json_file)
    json_file.close()
    reference_model = instantiate_model(
        model_json_path=settings["model_json_path"],
        weights_path=settings["weights_h5_path"],
        model_path=settings["model_h5_path"]
    )

    h5_db = h5py.File(config["db_path"], 'r')
    with open(config["split_path"], "r") as f:
        split = json.load(f)
        reference_list = split[args.indexes_name]

    if "image_activation" in list(config):
        image_activation_operator = \
            CustomOutputActivation(**config["image_activation"])
    else:
        image_activation_operator = CustomOutputActivation(type="clip")
    image_activation = image_activation_operator.activation

    if "GT type" in list(config.keys()):
        gt_type = config["GT type"]
    else:
        gt_type = "bbox"

    if "explanation_def" in list(config.keys()):
        explanation_def = config["explanation_def"]
    else:
        explanation_def = "st - adv"
    g0_model_settings = config["gen_0_settings"]
    generator_0 = instantiate_model(
        model_path=g0_model_settings["model_h5_path"]
    )
    g1_model_settings = config["gen_1_settings"]
    generator_1 = instantiate_model(
        model_path=g1_model_settings["model_h5_path"]
    )

    for ref in reference_list:
        data = h5_db[f"data/{ref[0]}/{ref[1]}/data"][()]
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        if len(data.shape) == 3:
            data = data[0]
        if reference_model.input_shape[1] == 3:
            data = np.array([[data, data, data]])
        else:
            data = np.array([[data]])
        prediction = reference_model(data)
        label_prediction = tf.round(prediction)
        if label_prediction < 0.5:
            _, xa = generator_0(data)
            xa = image_activation(xa)
            _, xst = generator_0(xa)
            xst = image_activation(xst)
        elif label_prediction >= 0.5:
            _, xa = generator_1(data)
            xa = image_activation(xa)
            _, xst = generator_1(xa)
            xst = image_activation(xst)
        else:
            raise KeyError

        if explanation_def == "st - adv":
            heatmap_batch = np.abs(xst - xa)[:, 0]
        elif explanation_def == "ori - adv":
            heatmap_batch = np.abs(data - xa)[:, 0]
        else:
            raise NotImplementedError
        heatmap = np.mean(heatmap_batch, axis=0)
        adv_image = xa.numpy()[0, 0]
        st_image = xst.numpy()[0, 0]
        heatmap_binary = np.where(heatmap >= np.percentile(heatmap, args.threshold), 1, 0)
        vis_data = data[0, 0]

        if gt_type == "bbox":
            annotations = h5_db[f"data/{ref[0]}/{ref[1]}/label/localization"][()]
        elif gt_type == "mask":
            annotations = h5_db[f"data/{ref[0]}/{ref[1]}/label/segmentation"][()]
        else:
            annotations = None

        show_multi_visualization(
            [heatmap, None, None, None, None],
            [vis_data, vis_data, adv_image, st_image, heatmap_binary],
            [annotations, annotations, annotations, annotations, annotations]
        )

    h5_db.close()


if __name__ == '__main__':
    main()
