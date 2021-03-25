import json
import numpy as np
import h5py
from tqdm import tqdm


def prediction_specific_split_from_reference(
        database, reference_list, classifier, normalize=True
    ):
    indexes_0, indexes_1 = [], []
    input_shape = classifier.input_shape[1:]
    for reference in tqdm(reference_list):
        data = database[f"data/{reference[0]}/{reference[1]}/data"][()]
        if normalize:
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        if len(data.shape) == 3:
            data = data[0]
        if input_shape[0] == 3:
            data = np.array([[data, data, data]])
        elif input_shape[0] == 1:
            data = np.array([[data]])
        else:
            raise NotImplementedError
        pred = np.ravel(classifier(data))
        if pred > 0.5:
            indexes_1.append(reference)
        elif pred <= 0.5:
            indexes_0.append(reference)
        else:
            raise KeyError
    return indexes_0, indexes_1


def set_prediction_specific_split(
        classifier, db_path, split,
        train_name="train_indexes", val_name="val_indexes"):
    db = h5py.File(db_path, "r")
    train = split[train_name]
    val = split[val_name]
    train_0, train_1 = prediction_specific_split_from_reference(
        db, train, classifier, normalize=True)
    val_0, val_1 = prediction_specific_split_from_reference(
        db, val, classifier, normalize=True)
    db.close()
    return train_0, train_1, val_0, val_1


def write_specific_split_to_json_file(
        path, train_0, train_1, val_0, val_1):
    split = {
        "train_indexes_0": train_0,
        "train_indexes_1": train_1,
        "val_indexes_0": val_0,
        "val_indexes_1": val_1,
    }
    with open(path, "w") as f:
        json.dump(split, f, indent=2)