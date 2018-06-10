import os
import csv
import math
import numpy as np
import os.path as Path 
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
            where row must be None
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
        target_np = self._resize(asmgen, padding)

        self.asmop_np = target_np
        return target_np

    def _visual(self, name):
        asm_np, _ = self.vengine(name)
        (tdim1, tdim2, tdim3) = self.shape
        tdim2 = math.ceil(asm_np.size / (tdim1 * tdim3))
        
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
                            pass  
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
                            pass 
        else:
            raise ValueError("Invalid order = {}")

        return padding
        

def load_label(name) -> Dict[str,str]:
    label = {}
    with open(name, "rb") as f:
        fdata = csv.reader(f)
        for id_class in fdata[1:]:
            id_class_sp = id_class.split(',')
            label.update({
                id_class_sp[0] : id_class_sp[1]
            })

    return label


