"""
Dataset partition utilities.
Class Partition is a index provider, thus provide index of next sample in ndarray.
from https://github.com/Hong-Xiang/dxlearn/tree/master/src/python/dxl/learn/dataset
"""
from collections import UserDict
from typing import Dict, Iterable
from .resampling import get_resampling
import numpy as np



class Partitioner:
    def _get_original_indices(self, data_column):
        if isinstance(data_column.capacity, int):
            return range(data_column.capacity)
        elif isinstance(data_column.capacity, Dict):
            return data_column.category

    def _get_valid_indices(self, indicies):
        return indicies

    def partition(self, data_column: Iterable) -> Iterable:
        def valid_index_generator(data_column, indices):
            for i in indices:
                yield data_column[i]

        return valid_index_generator(
            data_column,
            self._get_valid_indices(self._get_original_indices(data_column)))

    def get_capacity(self, data_columns):
        raise NotImplementedError


# self add
class CrossValidatePartitioner(Partitioner):
    def __init__(self, nb_blocks, in_blocks, resampling=None):
        super().__init__()
        self._nb_blocks = nb_blocks
        if isinstance(in_blocks, int):
            in_blocks = [in_blocks]
        self._in_blocks = in_blocks
        self._resampling = resampling

    def _get_valid_indices(self, indices):
        result = []
        # indices is a category Dict
        if isinstance(indices, Dict):
            class_info = {}
            for k, v in indices.items():
                class_info.update({
                    k: self.make_valid_indices(v)
                })
            # resamping
            if self._resampling:
                result = get_resampling(class_info, self._resampling)
            else:
                for i in class_info.values():
                    result.extend(i)
                # shuffle
                np.random.shuffle(result)
        # indices is range like 
        else:
            result = self.make_valid_indices(indices)

        return tuple(result)

    def make_valid_indices(self, indices):
        result = []
        len_block = len(indices) // self._nb_blocks
        for b in self._in_blocks:
            result += [
                indices[i] for i in range(b * len_block, (b + 1) * len_block)
            ]
        return result

    def get_capacity(self, data_columns):
        partition_capacity = []

        if isinstance(data_columns.capacity, int):
            len_block = data_columns.capacity // self._nb_blocks
            partition_capacity =  len_block * len(self._in_blocks)
        else:
            for _, v in data_columns.category.items():
                if self._resampling:
                    partition_capacity += self._resampling
                else:
                    kc = (len(v) // self._nb_blocks) * len(self._in_blocks)
                    partition_capacity += kc

        return partition_capacity
        


