import json
import numpy as np
import configargparse
import h5py
import tensorflow as tf
import os
from functools import partial
from tqdm import tqdm
from src.utils.evaluation_functions import load_json_param
from src.utils.instantiate_model import instantiate_model
from src.utils.custom_outputs_activation import CustomOutputActivation
from src.utils.perturbation import blur_perturbation


def main():
    p = configargparse.ArgParser()
    p.add_argument('--config-file', required=True)
    p.add_argument('--ref-model-settings-json-path', required=False)
    p.add_argument('--indexes-name', required=False, default="test_indexes")
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

    experiment_dir = config["experiment_dir"]
    explanation_path = os.path.join(experiment_dir, config["explanation_file_base_name"] + ".h5")
    explanation_file = h5py.File(explanation_path, 'w')
    if "data" not in list(explanation_file):
        explanation_file.create_group("data")

    h5_db = h5py.File(config["db_path"], 'r')
    with open(config["split_path"], "r") as f:
        split = json.load(f)
        reference_list = split[args.indexes_name]

    g0_model_settings = config["gen_0_settings"]
    generator_0 = instantiate_model(
        model_path=g0_model_settings["model_h5_path"]
    )
    g1_model_settings = config["gen_1_settings"]
    generator_1 = instantiate_model(
        model_path=g1_model_settings["model_h5_path"]
    )
    if "image_activation" in list(config):
        image_activation_operator = \
            CustomOutputActivation(**config["image_activation"])
    else:
        image_activation_operator = CustomOutputActivation(type="clip")
    image_activation = image_activation_operator.activation

    g0_cyc_settings = config["gen_0_cycgan_settings"]
    generator_01 = instantiate_model(
        model_path=g0_cyc_settings["model_h5_path"]
    )
    g1_cyc_settings = config["gen_1_cycgan_settings"]
    generator_10 = instantiate_model(
        model_path=g1_cyc_settings["model_h5_path"]
    )
    if "image_activation_cycgan" in list(config):
        image_activation_operator_cyc = \
            CustomOutputActivation(**config["image_activation_cycgan"])
    else:
        image_activation_operator_cyc = CustomOutputActivation(type="clip")
    image_activation_cyc = image_activation_operator_cyc.activation

    g_sa_settings = config["gen_sa_settings"]
    sagen = instantiate_model(
        model_path=g_sa_settings["model_h5_path"]
    )
    if "image_activation_sa" in list(config):
        image_activation_operator_sa = \
            CustomOutputActivation(**config["image_activation_sa"])
    else:
        image_activation_operator_sa = CustomOutputActivation(type="clip")
    image_activation_sa = image_activation_operator_sa.activation

    g_mg_settings = config["gen_mg_settings"]
    mgen = instantiate_model(
        model_path=g_mg_settings["model_h5_path"]
    )
    perturbation_func = partial(blur_perturbation, **config["perturbation_params"])
    for ref in tqdm(reference_list):
        label = h5_db[f"data/{ref[0]}/{ref[1]}/label/classification"][()]
        data = h5_db[f"data/{ref[0]}/{ref[1]}/data"][()]
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        if len(data.shape) == 3:
            data = data[0]
        data_full_p = perturbation_func(np.array([[data]]))
        if reference_model.input_shape[1] == 3:
            data = np.array([[data, data, data]])
            data_full_p = tf.concat([data_full_p, data_full_p, data_full_p], axis=1)
        else:
            data = np.array([[data]])
        prediction = reference_model(data)
        label_prediction = tf.round(prediction)

        mask = mgen([data, prediction])
        factor = int(np.array(data.shape[2]) / np.array(mask.shape[2]))
        if factor > 1:
            factor = (factor, factor)
            mask = tf.keras.layers.UpSampling2D(size=factor, data_format="channels_first")(mask)
        mask = tf.clip_by_value(mask, 0, 1)

        x_mg_a = data * (1 - mask) + data_full_p * mask
        x_sa_st, x_sa_a = sagen(data)
        x_sa_a = image_activation_sa(x_sa_a)
        x_sa_st = image_activation_sa(x_sa_st)
        if label_prediction < 0.5:
            _, xa = generator_0(data)
            xa = image_activation(xa)
            _, xst = generator_0(xa)
            xst = image_activation(xst)
            _, x_cyc_a = generator_01(data)
            x_cyc_a = image_activation_cyc(x_cyc_a)
        elif label_prediction >= 0.5:
            _, xa = generator_1(data)
            xa = image_activation(xa)
            _, xst = generator_1(xa)
            xst = image_activation(xst)
            _, x_cyc_a = generator_10(data)
            x_cyc_a = image_activation_cyc(x_cyc_a)
        else:
            raise KeyError

        group = f"data/{ref[0]}/{ref[1]}"

        explanation_file.create_dataset(group + "/x_real/data", data=data[0, 0])
        explanation_file.create_dataset(group + "/x_ours_a/data", data=xa.numpy()[0, 0])
        explanation_file.create_dataset(group + "/x_ours_st/data", data=xst.numpy()[0, 0])
        explanation_file.create_dataset(group + "/x_cyc_a/data", data=x_cyc_a.numpy()[0, 0])
        explanation_file.create_dataset(group + "/x_mg_a/data", data=x_mg_a.numpy()[0, 0])
        explanation_file.create_dataset(group + "/x_sa_a/data", data=x_sa_a.numpy()[0, 0])
        explanation_file.create_dataset(group + "/x_sa_st/data", data=x_sa_st.numpy()[0, 0])
        explanation_file.create_dataset(group + "/x_real/label", data=label)
        explanation_file.create_dataset(group + "/x_real/prediction", data=np.ravel(prediction))

    h5_db.close()
    explanation_file.close()


if __name__ == '__main__':
    main()
