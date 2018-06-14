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

    def test_capacity_category(self):
        datapath, nodepath, nb = self.get_or_create_dataset()
        a_datacolumn = UnbalancedNotFixedPyTablesColums(datapath, nodepath)

        _capacity = a_datacolumn.capacity
        assert isinstance(_capacity, Dict)
        cp = 0
        for k, v in _capacity.items():
            cp += v 
        assert cp == nb

        _category = a_datacolumn.category
        assert isinstance(_category, Dict)
        ct = 0
        label = ['0', '1']
        for k, v in _category.items():
            assert k in label
            ct += len(v)
        assert ct == nb
        a_datacolumn.close()

    def test_partition(self):
        nb_blocks = 
        datapath, nodepath, nb = self.get_or_create_dataset()
        partitioner = CrossValidatePartitioner()
        a_datacolumn = UnbalancedNotFixedPyTablesColums(datapath, nodepath)

    
if __name__ == "__main__":
    unittest.main()
