import os
import unittest
import pytest
import numpy as np
from zk.dataset.data_column import NestNPColumns
from zk.dataset.partitioner import CrossValidateResamplePartitioner

CURPATH = os.path.dirname(__file__)

class TestCrossValidateResamplePartitioner(unittest.TestCase):
    def get_path(self):
        '''
        nestnp:
            1:
                123.npy
                1234.npy
            2:
                111.npy
                2222.npy
                3333.npy
        '''
        return os.path.join(CURPATH, "nestnp")
    
    def get_columns(self):
        return NestNPColumns(self.get_path())

    def get_partitioner(self):
        return CrossValidateResamplePartitioner(
            3,
            [0, 1],
            {'1': 7, '2': 7}
        )

    def test_partition(self):
        partitioner = self.get_partitioner()
        p = partitioner.partition(self.get_columns())

        index = [i for i in p]
        assert len(index) == 8

    def test_capacity(self):
        partitioner = self.get_partitioner()
        c = partitioner.get_capacity(self.get_columns())

        assert c == 8