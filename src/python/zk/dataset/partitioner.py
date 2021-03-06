"""
Dataset partition utilities.
Class Partition is a index provider, thus provide index of next sample in ndarray.
redesigned from https://github.com/Hong-Xiang/dxlearn/tree/master/src/python/dxl/learn/dataset
"""
from typing import Dict, Iterable
from .resampling import get_resampling
import numpy as np


class Partitioner:
    def _get_original_indices(self, data_column):
        if isinstance(data_column.capacity, Dict):
            return data_column.category
        else:
            return range(data_column.capacity)

    def _get_valid_indices(self, indicies):
        return indicies

    def partition(self, data_column: Iterable) -> Iterable:
        def valid_index_generator(data_column, indices):
            for i in indices:
                # yield data_column[i]
                yield i

        return valid_index_generator(
            data_column,
            self._get_valid_indices(self._get_original_indices(data_column)))

    def get_capacity(self, data_columns):
        raise NotImplementedError


class CrossValidatePartitioner(Partitioner):
    def __init__(self, nb_blocks, in_blocks):
        super().__init__()
        self._nb_blocks = nb_blocks
        if isinstance(in_blocks, int):
            in_blocks = [in_blocks]
        self._in_blocks = in_blocks

    def _get_valid_indices(self, indices):
        result = []
        len_block = len(indices) // self._nb_blocks
        for b in self._in_blocks:
            result += [
                indices[i] for i in range(b * len_block, (b + 1) * len_block)
            ]
        return tuple(result)

    def get_capacity(self, data_columns):
        len_block = data_columns.capacity // self._nb_blocks
        return len_block * len(self._in_blocks)

# self add
class CrossValidateResamplePartitioner(Partitioner):
    def __init__(self, nb_blocks, in_blocks, resampling=None):
        super().__init__()
        self._nb_blocks = nb_blocks
        if isinstance(in_blocks, int):
            in_blocks = [in_blocks]
        self._in_blocks = in_blocks
        self._resampling = resampling or {}

    def _get_valid_indices(self, indices):
        result = []
        # indices is a category Dict
        if isinstance(indices, Dict):
            # resamping
            if self._resampling:
                resample_result = get_resampling(indices, self._resampling)
            else:
                resample_result = indices

            result = []
            for k, v in resample_result.items():
                result.extend(self.make_valid_indices(v))
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
        partition_capacity = 0

        if isinstance(data_columns.capacity, Dict):
            for k, v in data_columns.category.items():
                if k in self._resampling:
                    lg = self._resampling[k]
                else:
                    lg = len(v)

                kc = (lg // self._nb_blocks) * len(self._in_blocks)
                partition_capacity += kc
        else:
            len_block = data_columns.capacity // self._nb_blocks
            partition_capacity =  len_block * len(self._in_blocks)

        return partition_capacity
        


