import os
import unittest
import tables as tb
import numpy as np
from keras.utils import to_categorical 
from zk.dataset.dataset import DataSet
from zk.dataset.partitioner import CrossValidatePartitioner
from zk.dataset.data_column import PyTablesColumns

CURPATH = os.path.dirname(__file__)

class TestDataSet(unittest.TestCase):
    class CONFING:
        BATCH_SIZE = 5
        SHAPE = (2, 10, 8)
        NB_BLOCKS = 5
        IN_BLOCKS = [1, 2, 3]
        NB_CLASS = 2
        NB_DATA = 100
        
    def get_or_create_h5(self):
        name = "fixed.h5"
        nodepath = "/mcc2015/data"
        shape = self.CONFING.SHAPE
        nb_class = self.CONFING.NB_CLASS
        nb = self.CONFING.NB_DATA

        dataset = os.path.join(CURPATH, name)
        if not os.path.isfile(dataset):
            f5 = tb.open_file(dataset, "w")
            group = f5.create_group("/", "mcc2015")
            tb_desp = self.get_tb_desp(shape, nb_class)
            a_table = f5.create_table(group, "data", tb_desp)
            a_row = a_table.row

            label = [0, 0, 0, 1]
            for i in range(nb):
                y_label = label[i%4]
                x = np.ones(shape)
                 # fill table
                a_row['x'] = x
                a_row['y'] = to_categorical(y_label, 2)
                a_row.append()
                # flush table 
                a_table.flush()  
            f5.close()

        return dataset, nodepath, nb
    
    def get_tb_desp(self, shape, nb_class):
        class TbDesp(tb.IsDescription):
            x = tb.UInt8Col(shape=shape)
            y = tb.UInt8Col(shape=(nb_class,))
        return TbDesp

    def get_columns(self):
        dataset, nodepath, _ = self.get_or_create_h5()
        return PyTablesColumns(dataset, nodepath)

    def get_partition(self):
        C = self.CONFING
        return CrossValidatePartitioner(C.NB_BLOCKS, C.IN_BLOCKS)

    def make_dataset(self):
        C = self.CONFING
        return DataSet(self.get_columns(),
                       self.get_partition(),
                       C.BATCH_SIZE,
                       C.SHAPE,
                       C.NB_CLASS)

    def test_step_len(self):
        C = self.CONFING
        dataset = self.make_dataset()
        crossvaliddata = (C.NB_DATA // C.NB_BLOCKS) * len(C.IN_BLOCKS)
        assert len(dataset) == crossvaliddata // C.BATCH_SIZE

    def test_indexs(self):
        C = self.CONFING
        dataset = self.make_dataset()
        crossvaliddata = (C.NB_DATA // C.NB_BLOCKS) * len(C.IN_BLOCKS)
        assert len(dataset.indexs) == crossvaliddata

    def test_getitem(self):
        C = self.CONFING
        dataset = self.make_dataset()
        X, y = dataset[2]

        self.assertTupleEqual(X.shape, (C.BATCH_SIZE, *C.SHAPE))
        self.assertTupleEqual(y.shape, (C.BATCH_SIZE, C.NB_CLASS))
        
