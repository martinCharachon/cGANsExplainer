import numpy as np
import h5py
import json
from tqdm import tqdm
from .localization_utils import LocalisationTools


def mean_results(results_dict, name="all_iou", unique=False):
    metrics = []
    if unique:
        for mid, value in tqdm(results_dict.items()):
            result = results_dict[mid]
            metrics.append(result[name])
        mean_metrics = float(np.mean(metrics))
    else:
        for mid, value in tqdm(results_dict.items()):
            result = results_dict[mid]
            metrics.append(result[name])
        metrics = np.array(metrics)
        mean_metrics = np.mean(metrics, 0).tolist()
    print(f"Mean {name} = {mean_metrics}")
    print()
    results_dict[f"mean_{name}"] = mean_metrics
    return results_dict


def evaluate_metrics(
        config, explanation_path,
        metrics_name="all_iou", indexes_name="test_indexes", gt_type="bbox", unique=False):
    h5_db = h5py.File(config["db_path"], 'r')
    h5_explanation = h5py.File(explanation_path, 'r')
    with open(config["split_path"], "r") as f:
        split = json.load(f)
    reference_list = split[indexes_name]
    loc_tool = LocalisationTools([metrics_name])
    results_dict = {}
    for (i, ref) in enumerate(reference_list):
        label = h5_db[f"data/{ref[0]}/{ref[1]}/label/classification"][()]
        if label != 1:
            continue
        if gt_type == "bbox":
            annotation = h5_db[f"data/{ref[0]}/{ref[1]}/label/localization"][()]
        elif gt_type == "mask":
            annotation = h5_db[f"data/{ref[0]}/{ref[1]}/label/segmentation"][()]
        else:
            raise AttributeError

        explanation = h5_explanation[f"data/{ref[0]}/{ref[1]}/data"][()]

        res = loc_tool.compute_metrics(explanation, annotation)
        m = res[metrics_name]
        if ref[1] != 0:
            results_dict[str(ref)] = {
                metrics_name: m
            }

    final_results_dict = mean_results(
        results_dict, name=metrics_name, unique=unique)

    results_path = f'{config["results_path"]}_{metrics_name}.json'
    with open(results_path, "w") as f:
        json.dump(final_results_dict, f, indent=2)
    h5_explanation.close()
    h5_db.close()


def add_interpretation_and_infos_to_explanation_file(
        h5file,
        reference,
        heatmap,
        adv_image,
        st_image=None,
        additional_infos=None
):
    group = f"data/{reference[0]}/{reference[1]}"
    if reference[0] in list(h5file["data"]):
        del h5file[group]
    h5file.create_dataset(group + "/data", data=heatmap, compression="gzip")
    h5file.create_dataset(group + "/adv", data=adv_image, compression="gzip")
    if st_image is not None:
        h5file.create_dataset(group + "/st", data=st_image, compression="gzip")
    if additional_infos is not None:
        h5file.create_dataset(group + "/label/prediction", data=additional_infos[0])


def load_json_param(json_path):
    if json_path is None:
        return None
    try:
        json_file = open(json_path, 'r')
        params = json.load(json_file)
        json_file.close()
    except Exception:
        params = dict()
    return params