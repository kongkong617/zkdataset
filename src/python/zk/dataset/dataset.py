"""
from https://github.com/Hong-Xiang/dxlearn/tree/master/src/python/dxl/learn/dataset
"""
import tables as tb
from typing import Dict
import tensorflow as tf


class DatasetFromColumns:
    def __init__(self,
                 columns,
                 nb_epochs,
                 batch_size=None,
                 shuffle=True):
        self.nb_epochs = nb_epochs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._columns = columns
        self._data = []
        self._iterator = None

    def _batch_iterator(self):
        if self.shuffle:
            nb_fetch = 4 * self.batch_size
            for i in range(nb_fetch):
                self._data[i] = next(self._columns)

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self)
        return next(self._iterator)

    def __iter__(self):
        return self._batch_iterator()

    
        

    

