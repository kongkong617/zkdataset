import os
import unittest
import numpy as np
import tables as tb
from typing import Dict
from keras.utils import to_categorical 
from zk.dataset.partitioner import CrossValidatePartitioner
from zk.dataset.data_column import UnbalancedNotFixedPyTablesColums

CURPATH = os.path.dirname(__file__)

class TestUnbalancedNotFixedPyTablesColums(unittest.TestCase):
    def get_or_create_dataset(self):
        name = "unbalancenotfixed.h5"
        nb = 40

        dataset = os.path.join(CURPATH, name)
        if not os.path.isfile(dataset):
            f5 = tb.open_file(dataset, "w")
            group = f5.create_group("/", "mcc2015")

            label = [0, 0, 0, 1]
            for i in range(nb):
                y_label = label[i%4]
                x = np.ones([8, i+2, 10])
                x_shape = x.shape
                # create tabel
                t_name = "L_" + str(y_label) + "_" + str(i)
                tb_desp = self.get_tb_desp(x_shape)
                a_table = f5.create_table(group, t_name, tb_desp)
                a_row = a_table.row
                 # fill table
                a_row['x'] = x
                a_row['y'] = to_categorical(y_label, 2)
                a_row.append()
                # flush table 
                a_table.flush()  
            f5.close()

        nodepath = "/mcc2015"
        return dataset, nodepath, nb
    
    def get_tb_desp(self, x_shape):
        class TbDesp(tb.IsDescription):
            x = tb.UInt8Col(shape=x_shape)
            y = tb.UInt8Col(shape=(2,))
        return TbDesp

    def test_capacity(self):
        datapath, nodepath, nb = self.get_or_create_dataset()
        a_datacolumn = UnbalancedNotFixedPyTablesColums(datapath, nodepath)

        _capacity = a_datacolumn.capacity
        assert isinstance(_capacity, Dict)
        cp = 0
        for k, v in _capacity.items():
            cp += v 
        assert cp == nb
        a_datacolumn.close()

    def test_category(self):
        datapath, nodepath, nb = self.get_or_create_dataset()
        a_datacolumn = UnbalancedNotFixedPyTablesColums(datapath, nodepath)

        _category = a_datacolumn.category
        assert isinstance(_category, Dict)
        ct = 0
        label = ['0', '1']
        for k, v in _category.items():
            if k == label[0]:
                assert len(v) == (nb // 4) * 3
            else:
                assert len(v) == nb // 4
        a_datacolumn.close()

    def test_partition(self):
        nb_blocks = 5
        in_block = [0, 1, 2]

        datapath, nodepath, nb = self.get_or_create_dataset()
        partitioner = CrossValidatePartitioner(nb_blocks, in_block)
        a_datacolumn = UnbalancedNotFixedPyTablesColums(datapath, nodepath, partitioner)

        count = 0
        for x, y in a_datacolumn:
            assert x.shape[1] == 8
            assert y.shape[0] == 2
            count += 1
        assert count == (nb // nb_blocks) * len(in_block)
        a_datacolumn.close()

    def test_partition_resampling(self):
        nb_blocks = 5
        in_block = [0, 1, 2]
        resampling = 12

        datapath, nodepath, nb = self.get_or_create_dataset()
        partitioner = CrossValidatePartitioner(nb_blocks, in_block, resampling)
        a_datacolumn = UnbalancedNotFixedPyTablesColums(datapath, nodepath, partitioner)

        count = 0
        for _, _ in a_datacolumn:
            count += 1
        assert count == resampling * 2
        a_datacolumn.close()

if __name__ == "__main__":
    unittest.main()