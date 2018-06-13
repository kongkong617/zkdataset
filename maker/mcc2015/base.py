import os
import numpy as np  
import tables as tb
from typing import Dict 
from keras.utils import to_categorical


class DirDataGenerator:
    def __init__(self, name):
        self._g = self._make_gen(name)

    def _make_gen(self, name):
        for dirpath, _, filenames in os.walk(name):
            if filenames:
                label = os.path.basename(dirpath)
                for name in filenames:
                    file_path = os.path.join(dirpath, name)
                    data = np.load(file_path)

                    yield label, name, data  

    def __iter__(self):
        return self._g

    def __next__(self):
        return next(self._g)


class MccPytablesMaker:
    def __init__(self, name, nb_class, data_gen):
        self.name = name
        self.nb_class = nb_class
        self.data_gen = data_gen

    def make(self):
        f5 = tb.open_file(self.name, "w")
        group = f5.create_group("/", "mcc2015")

        for (label, name, data) in self.data_gen:
            x_shape = data.shape
            name = label + "_" + name
            # create tabel
            print("create table {}".format(name))
            tb_desp = self.make_tb_desp(x_shape)
            a_table = f5.create_table(group, name, tb_desp)
            a_row = a_table.row
            # fill table
            a_row['x'] = data
            a_row['y'] = to_categorical(int(label), self.nb_class)
            # flush table 
            a_table.flush()
        
        self.handl = f5
    
    @property
    def capacity(self):
        root = self.handl.get_node('/')
        nb_nodes = len(root._f_iter_nodes())

        return nb_nodes

    def close(self):
        self.handl.close()

    def make_tb_desp(self, x_shape):
        class TbDesp(tb.IsDescription):
            x = tb.UInt8Col(shape=x_shape)
            y = tb.UInt8Col(shape=(self.nb_class,))

        return TbDesp












