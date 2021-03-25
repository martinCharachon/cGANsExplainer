import numpy as np
import h5py
from .data_preprocessor import IdentityPreprocessor, set_geometric_preprocessor


def set_data_generator(
        input_shape, db_path, reference_list,
        batch_size=1, preprocessor_config=None, seed=0):
    manager = DataGeneratorFromH5File(
        input_shape=tuple(input_shape),
        path_to_hdf5_db=db_path,
        reference_list=reference_list,
        batch_size=batch_size,
        do_shuffle=True,
        seed=seed,
        preprocessor=set_geometric_preprocessor(preprocessor_config)
    )
    return manager


class DataGeneratorFromH5File:

    def __init__(self,
                 input_shape,
                 path_to_hdf5_db,
                 reference_list,
                 preprocessor=IdentityPreprocessor(),
                 batch_size=1,
                 seed=0,
                 passes=np.inf,
                 do_shuffle=False):
        np.random.seed(seed)
        self._db = h5py.File(path_to_hdf5_db, "r")
        self._reference_list = reference_list
        self.num_index = len(reference_list)
        self._index_list = self._set_index_list()
        self._batch_size = batch_size
        self.passes = passes
        self._input_shape = input_shape
        self._preprocessor = preprocessor
        self._do_shuffle = do_shuffle

    def _set_index_list(self):
        return np.arange(0, self.num_index)

    def _preprocess_data(self, data):
        return self._preprocessor(data, target_size=self._input_shape)

    def _generator(self):
        epochs = 0
        while epochs < self.passes:
            self._shuffle_index_list()
            for index in np.arange(0, self.num_index, self._batch_size):
                x_list = []
                y_list = []
                for i in self._index_list[index:index + self._batch_size]:
                    ref = self._reference_list[i]
                    new_data = self._get_data(ref)
                    new_data = self._reformat_input_for_keras_model(new_data)
                    new_data = self._preprocess_data(new_data)
                    new_annotation = self._get_annotation(ref)
                    x_list.append(new_data)
                    y_list.append(new_annotation)
                x_batch = np.array(x_list)
                y_batch = np.array(y_list)
                yield x_batch, y_batch
            epochs += 1

    def _reformat_input_for_keras_model(self, x):
        if len(x.shape) == 3:
            x = x[0]
        if self._input_shape[0] == 3:
            return np.array([x, x, x])
        elif self._input_shape[0] == 1:
            return np.array([x])
        else:
            raise NotImplementedError

    def _shuffle_index_list(self):
        if self._do_shuffle:
            np.random.shuffle(self._index_list)

    def _get_data(self, reference):
        return self._db[f"data/{reference[0]}/{reference[1]}/data"][()]

    def close(self):
        self._db.close()

    def _get_annotation(self, reference):
        return self._db[f"data/{reference[0]}/{reference[1]}/label/classification"][()]

    def __call__(self, *args, **kwargs):
        return self._generator()
