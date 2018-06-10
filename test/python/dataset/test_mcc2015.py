import os
import unittest
import numpy as np
from zk.dataset.mcc2015 import AsmopcodeData, AsmopGenerator, load_label
from zk.visual import Visual

CURPATH = os.path.dirname(__file__)

class TestMcc2015(unittest.TestCase):
    def test_AsmopGenerator(self):
        name001 = os.path.join(CURPATH, "asm001")
        asm_np, _ = Visual(name=name001,
                            vtype=3,
                            vwidth=64)()
        asmgen = AsmopGenerator(asm_np)
        _opcode19 = "56"
        _opcode19visul = [0, 255, 0, 255, 0, 255, 255, 0] + [0 for i in range(56)]
        _opcode19visulnp = np.array(_opcode19visul)

        asm_list = [i for i in asmgen]
        self.assertEqual(len(asm_list), 22)
        self.assertEqual(asm_list[19].all(), _opcode19visulnp.all())

    def test_AsmopcodeData_shape(self):
        name001 = os.path.join(CURPATH, "asm001")
        asm_resizer = AsmopcodeData(name=name001,
                                   vencodelen=64,
                                   shape=(8, None, 128),
                                   order=(1, 3, 2))()
        _tshape = (8, 2, 128)
        self.assertTupleEqual(asm_resizer.shape, _tshape)

    def test_AsmopcodeData_order(self):
        name001 = os.path.join(CURPATH, "asm001")
        asm_resizer = AsmopcodeData(name=name001,
                                   vencodelen=64,
                                   shape=(8, None, 128),
                                   order=(1, 3, 2))()
        _opcode19visul = [0, 255, 0, 255, 0, 255, 255, 0] + [0 for i in range(56)]
        _opcode19visulnp = np.array(_opcode19visul)
        opcode19visulnp = asm_resizer[3, 1, 0:64]
        self.assertEqual(opcode19visulnp.all(), _opcode19visulnp.all())

    def test_load_label(self):
        name = os.path.join(CURPATH, "trainLabels.csv")
        label = load_label(name)

        _key = "0rgudc7PpbexCtBjNqWF"
        self.assertEqual(len(label), 38)
        self.assertEqual(label[_key], 2)

if __name__ == "__main__":
    unittest.main()