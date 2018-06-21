import keras
import tables as tb
import numpy as np
from typing import Tuple


class DataSet(keras.utils.Sequence):
    class KEYS:
        X = 'x'
        Y = 'y'

    def __init__(self, columns, partitioner, batch_size, shape, nb_class, shuffle=True):
        self.columns = columns
        self.partitioner = partitioner
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shape = shape
        self.nb_class = nb_class

        self.indexs = np.array([i for i in partitioner.partition(columns)])
        np.random.shuffle(self.indexs)

    def __len__(self):
        return self.partitioner.get_capacity(self.columns) // self.batch_size

    def __getitem__(self, index):
        indexs = self.indexs[index*self.batch_size : (index+1)*self.batch_size]
        X, y = self._data_generation(indexs)
        return X, y

    def _data_generation(self, list_ids):
        X = np.empty((self.batch_size, *self.shape))
        y = np.empty((self.batch_size, self.nb_class))

        for i, id in enumerate(list_ids):
            result = self.columns[id]
            X[i, ] = result[self.KEYS.X]
            y[i, ] = result[self.KEYS.Y]

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexs)
        

    
        

    

