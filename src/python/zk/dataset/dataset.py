"""
from https://github.com/Hong-Xiang/dxlearn/tree/master/src/python/dxl/learn/dataset
"""
import tables as tb
from typing import Dict
import tensorflow as tf


class Dataset:
    class KEYS:
        class CONFIG:
            NB_EPOCHS = 'nb_epochs'
            BATCH_SIZE = 'batch_size'
            IS_SHUFFLE = 'is_shuffle'

    def __init__(self,
                 nb_epochs=None,
                 batch_size=None,
                 is_shuffle=None):

        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle

    def _process_dataset(self, dataset):
        dataset = dataset.repeat(self.nb_epochs)
        if self.is_shuffle:
            dataset = dataset.shuffle(self.batch_size * 4)
        dataset = dataset.batch(self.batch_size)
        return dataset


class DatasetFromColumns(Dataset):
    class KEYS:
        class TENSOR:
            DATA = 'data'

    def __init__(self,
                 columns,
                 nb_epochs=None,
                 batch_size=None,
                 is_shuffle=None):
        super().__init__(
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            is_shuffle=is_shuffle)

        self.tensors = {}
        self._columns = columns
        self.kernel()

    def _make_dataset_object(self):
        return tf.data.Dataset.from_generator(
            self._columns.__iter__, self._columns.types, self._columns.shapes)

    def _make_dataset_tensor(self, dataset):
        result = tf.Tensor(dataset.make_one_shot_iterator().get_next())
        if self.batch_size is not None:
            shape = result.data.shape.as_list()
            shape[0] = self.batch_size
            result = tf.Tensor(tf.reshape(result.data, shape))
        return result

    def kernel(self, inputs=None):
        dataset = self._make_dataset_object()
        dataset = self._process_dataset(dataset)
        self.tensors[self.KEYS.TENSOR.DATA] = self._make_dataset_tensor(
            dataset)

