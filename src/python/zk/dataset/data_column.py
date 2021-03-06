"""
DataColumns, a representation of table-like data.
redesigned from https://github.com/Hong-Xiang/dxlearn/tree/master/src/python/dxl/learn/dataset
"""
import os
import h5py
import tables as tb
import numpy as np
from typing import Dict, Iterable


class DataColumns:
    def __init__(self, data):
        self._category_cache = {}
        self._capacity_cache = None
        self._iterator = None
        self._dataset_nodes = None
        self.data = self._process(data)

    def _process(self, data):
        return data

    @property
    def columns(self):
        return self.data.keys()

    def _calculate_capacity(self):
        raise NotImplementedError

    def _calculate_category(self):
        raise NotImplementedError

    @property
    def capacity(self):
        if self._capacity_cache is not None:
            return self._capacity_cache
        else:
            return self._calculate_capacity()

    @property
    def category(self):
        if self._category_cache is not None:
            return self._category_cache
        else:
            return self._calculate_category()

    @property
    def shapes(self):
        raise NotImplementedError

    @property
    def types(self):
        raise NotImplementedError

    def _make_iterator(self):
        raise NotImplementedError

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self)
        return next(self._iterator)

    def __iter__(self):
        return self._make_iterator()


class DataColumnsWithGetItem(DataColumns):
    def _make_iterator(self):
        def it():
            for i in range(self.capacity):
                yield self.__getitem__(i)

        return it()

    def __getitem__(self, i):
        raise NotImplementedError


class PyTablesColumns(DataColumnsWithGetItem):
    def __init__(self, path_file, path_dataset):
        super().__init__((path_file, path_dataset))

    def _process(self, data):
        path_file, path_dataset = data
        self._file = tb.open_file(str(path_file))
        self._node = self._file.get_node(path_dataset)

    def __getitem__(self, i):
        result = {}
        data = self._node[i]
        for k in self.columns:
            result[k] = np.array(data[k])
        return result

    @property
    def columns(self):
        return tuple(self._node.colnames)

    def _calculate_capacity(self):
        return self._node.shape[0]

    def close(self):
        self._file.close()


class NestNPColumns(DataColumns):
    def __init__(self, path):
        super().__init__(path)

    def _process(self, data):
        for dirpath, _, filenames in os.walk(data):
            if filenames:
                index = []
                label = os.path.basename(dirpath)
                for d in filenames:
                    index.append(os.path.join(dirpath, d))
                self._category_cache.update({label: index})

        return self._category_cache
    
    def __getitem__(self, i):
        label = os.path.basename(os.path.dirname(i))
        return {
            'x' : np.load(i),
            'y' : int(label) - 1
        }

    def _calculate_capacity(self):
        _capacity = {}
        for k, v in self._category_cache.items():
            _capacity.update({k : len(v)})

        self._capacity_cache = _capacity
        return _capacity
    
    def _make_iterator(self):
        def it():
            for _, v in self.category.items():
                for i in v:
                    yield self.__getitem__(i)
        
        return it()


# self add 
class UnbalancedNotFixedPyTablesColums(DataColumns):
    """
    Unbalanced:
        Each class has different numbers of samples.
            so need to be re-sampling to same
    NotFixed:
        Each sample has a different shape, 
            then make a sample a dataset and let all datasets in a node.

            The dataset name must be encode like:
                L_1_agltS1qfOZiUvzBy24uc
            where:
                `L` is a meaningless letter, just for escape namewarrning
                `_` is a split mark
                `1` is an sample class label
                `agltS1qfOZiUvzBy24uc` is an unique sample name

            The dataset colnames must be encode like:
                x = .... , represent data
                y = .... , represent label
    """
    class KEYS:
        MARK = "_"
        COLNAME_X = 'x'
        COLNAME_Y = 'y'

    def __init__(self, path_file, path_dataset, partitioner=None, unlimited=False):
        super().__init__((path_file, path_dataset))
        self._unlimited = unlimited
        self._partitioner = partitioner

    def _process(self, data):
        path_file, path_dataset = data
        self._file = tb.open_file(path_file, "r")
        self._node = self._file.get_node(path_dataset)

        self._dataset_nodes = self._node._f_list_nodes()

        for i, dataset in enumerate(self._dataset_nodes):
            label = dataset.name.split(self.KEYS.MARK)[1]
            index = self._category_cache.get(label)
            if index is None:
                index = []
            
            index.append(i)
            self._category_cache.update({label : index})

    def _calculate_capacity(self):
        _capacity = {}
        for k, v in self._category_cache.items():
            _capacity.update({k : len(v)})

        self._capacity_cache = _capacity
        return _capacity

    def __getitem__(self, i):
        node = self._dataset_nodes[i]
        colnames = node.colnames
        # print(i)
        # print(colnames)
        x = node[0][self.KEYS.COLNAME_X]
        y = node[0][self.KEYS.COLNAME_Y]

        y = np.array(y)
        # print(y.shape)
        # y = np.reshape(y, [-1, 9])

        if len(x.shape) == 3:
            shape = [-1] + [i for i in x.shape]
            x = np.reshape(x, shape)
        x = np.array(x)
        # print(x.shape)

        return (x, y)

    def _make_iterator(self):
        return self._partitioner.partition(self, self._unlimited)

    @property
    def columns(self):
        return [self.KEYS.COLNAME_X, self.KEYS.COLNAME_Y]

    def close(self):
        self._file.close()