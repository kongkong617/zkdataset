import os
import csv
import math
import random
import numpy as np
import os.path as Path
import tables as tb
from keras.utils import to_categorical 
from typing import Dict, List, Tuple
from zk.visual import Visual

LABEL = {
    '1': 'Ramnit',
    '2': 'Lollipop',
    '3': 'Kelihos_ver3',
    '4': 'Vundo',
    '5': 'Simda',
    '6': 'Tracur',
    '7': 'Kelihos_ver1',
    '8': 'Obfuscator.ACY',
    '9': 'Gatak'
}

QUANTITY = {
    '1': 1541,
    '2': 2478,
    '3': 2942,
    '4': 475,
    '5': 42,
    '6': 751,
    '7': 398,
    '8': 1220,
    '9': 1013
}

class AsmopGenerator:
    """
    Argument:
        `asmop_np`: A asmopcode visual numpy.
            row: number of asmopcode
            col: visual coding length of one asmopcode
    Return:
        A generator, put a row in asmop_np each time
    """
    def __init__(self, asmop_np):
        self.asmop_np = asmop_np
        self.nb_opcode = asmop_np.shape[0]
        self._g = self._make_generator()

    def __iter__(self):
        return self 

    def __next__(self):
        return next(self._g)

    def _make_generator(self):
        for i in range(self.nb_opcode):
            yield self.asmop_np[i, :]


class AsmopcodeData:
    """
    Argument:
        `name`: os.path
        `vencodelen`: A Integer
            viusal encoding length,  one asmopcode encoding `vlencodelen` bits
        `shape`: A tuple of visual asmopcode shape
            (dim1, dim2, dim3)
            it could be (channel, row, col) or (row, col, channel)
            if row == None, row is unfixed.
            else row is fixed.
        `order`: A tuple of index order, who padding first.
            it must be one of:
            >>> (1, 3, 2)   (channel, row, col)
            dim0 padding first, then dim3, finally dim2 
            >>> (3, 2, 1)   (row, col, channel)
            dim3 padding first, then dim2, finally dim1
    """
    def __init__(self,
                 name: Path,
                 vencodelen:int,
                 shape: Tuple,
                 order: Tuple):
        self.name = name
        self.vencodelen = vencodelen
        self.shape =  shape
        self.order = order
        self.vengine = Visual(name=name, vtype=3, vwidth=vencodelen)
        self.asmop_np = self.construct(name, True)

    def __call__(self, name=None):
        return self.construct(name, False)

    def construct(self, name, is_create):
        if is_create and name == None:
            return None
        elif not is_create and name == None:
            return self.asmop_np

        asmgen, padding = self._visual(name)
        if padding is None:     # fix dim2==0
            return None

        target_np = self._resize(asmgen, padding)

        self.asmop_np = target_np
        return target_np

    def _visual(self, name):
        asm_np, _ = self.vengine(name)
        (tdim1, tdim2, tdim3) = self.shape

        if self.order == (1, 3, 2):
            cal_tdim2 = math.ceil(asm_np.size / (tdim1 * tdim3))
            if not tdim2:
                tdim2 = cal_tdim2
            
        
        if self.order == (3, 2, 1):
            cal_tdim1 = math.ceil(asm_np.size / (tdim2 * tdim3))
            if not tdim1:
                tdim1 = cal_tdim1

        if 0 in (tdim1, tdim2, tdim3):    # fix vengine return asm_np.size == 0
            return None, None
        
        padding = np.zeros([tdim1, tdim2, tdim3])
        asmgen = AsmopGenerator(asm_np)
        return asmgen, padding

    def _resize(self, asmgen, padding):
        (dim1, dim2, dim3) = padding.shape
        
        if tuple(self.order) == (1, 3, 2):  # (channel, row, col)
            n_col, check = divmod(dim3, self.vencodelen)
            if check:
                raise ValueError("Invalid shape[2] = {}".format(self.shape[2]))
            for i_row in range(dim2):
                for i_col in range(n_col):
                    start = i_col * self.vencodelen
                    end = start + self.vencodelen
                    for i_channel in range(dim1):
                        try:
                            padding[i_channel, i_row, start:end] = next(asmgen)
                        except StopIteration:
                            break  
        elif tuple(self.order) == (3, 2, 1):   # (row, col, channel)
            n_col, check = divmod(dim2, self.vencodelen)
            if check:
                raise ValueError("Invalid shape[1] = {}".format(self.shape[1]))
            for i_row in range(dim1):
                for i_col in range(n_col):
                    start = i_col * self.vencodelen
                    end = start + self.vencodelen
                    for i_channel in range(dim3):
                        try:
                            padding[i_row, start:end, i_channel] = next(asmgen)
                        except StopIteration:
                            break 
        else:
            raise ValueError("Invalid order = {}")

        return padding
        

def load_label(name:Path) -> Dict[str,str]:
    """
    Argumet:
        `name`: Path to a csv file
            which like:
            row0:   "Id","Class"
            row1:   "01kcPWA9K2BOxQeS5Rju",1
                        .   .   .   
                        .   .   .          
            rown:   "0ZiQmgtxzHe9v5O8Lf2k",m   
    """
    label = {}
    with open(name) as f:
        fdata = csv.reader(f)
        heading = next(fdata)
        for row in fdata:
            label.update({
                row[0]: int(row[1])
            })

    return label


class DirDataGenerator:
    def __init__(self, name):
        self._data = self._list_data(name)
        self._g = self._make_gen()

    def _list_data(self, name) -> List:
        _data = []
        for dirpath, _, filenames in os.walk(name):
            if filenames:
                for d in filenames:
                    _data.append(os.path.join(dirpath, d))

        random.shuffle(_data)
        return _data

    def _make_gen(self):
        for fpath in self._data:
            label = os.path.basename(os.path.dirname(fpath))
            data = np.load(fpath)
            name = os.path.basename(fpath).split(".")[0]
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
        self._summary = {}

    def make(self):
        f5 = tb.open_file(self.name, "w")
        group = f5.create_group("/", "mcc2015")

        for (label, name, data) in self.data_gen:
            if not self.filter_shape(data):
                continue

            if not self._summary.get(label):
                self._summary.update({label : 1})
            else:
                count = self._summary.get(label)
                self._summary.update({label : count+1})

            x_shape = data.shape
            name = "L_" + label + "_" + name
            # create tabel
            print("create table {}".format(name))
            tb_desp = self.make_tb_desp(x_shape)
            a_table = f5.create_table(group, name, tb_desp)
            a_row = a_table.row
            # fill table
            a_row['x'] = data
            a_row['y'] = to_categorical(int(label)-1, self.nb_class)
            a_row.append()
            # flush table 
            a_table.flush()
            # print("flush table {}".format(name))
        
        self.handl = f5
    
    @property
    def capacity(self):
        return self._summary

    def close(self):
        self.handl.close()

    def filter_shape(self, data):
        _shape = data.shape
        for i in _shape:
            if i <= 0:
                return False
        return True

    def make_tb_desp(self, x_shape):
        class TbDesp(tb.IsDescription):
            x = tb.UInt8Col(shape=x_shape)
            y = tb.UInt8Col(shape=(self.nb_class,))

        return TbDesp


class MccFixedPytablesMaker:
    def __init__(self, name, nb_class, shape, data_gen):
        self.name = name
        self.nb_class = nb_class
        self.shape = shape
        self.data_gen = data_gen
        self._summary = {}

    def make(self):
        f5 = tb.open_file(self.name, "w")
        group = f5.create_group("/", "mcc2015")
        # create tabel
        tb_desp = self.make_tb_desp()
        table = f5.create_table(group, "data", tb_desp)
        a_row = table.row

        count = 0
        for (label, name, data) in self.data_gen:
            if not self._summary.get(label):
                self._summary.update({label : 1})
            else:
                count = self._summary.get(label)
                self._summary.update({label : count+1})
            # fill table
            a_row['x'] = data
            # a_row['y'] = to_categorical(int(label)-1, self.nb_class)
            a_row['y'] = int(label) - 1
            a_row.append()
            # flush table 
            count += 1
            if count % 500 == 0:
                table.flush()
                print("flush table {}".format(count))

        f5.close()
    
    @property
    def capacity(self):
        return self._summary

    def make_tb_desp(self):
        class TbDesp(tb.IsDescription):
            x = tb.UInt8Col(shape=self.shape)
            y = tb.UInt8Col(shape=())

        return TbDesp