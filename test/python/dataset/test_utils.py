import os
import unittest
from zk.dataset.utils import instruction_count, instruction_length

CURPATH = os.path.dirname(__file__)

class TestUtils(unittest.TestCase):
    def test_instrcution_count(self):
        a = ".text:00401019 CC CC CC CC CC CC CC					       align 10h"
        expect_count_a = 56
        assert expect_count_a == instruction_length(a)

    def test_instrcution_length(self):
        name = os.path.join(CURPATH, 'asm001')
        expect_count = 22
        count, lg = instruction_count(name)

        assert count == expect_count
        assert lg[8] == 5
        assert lg[16] == 4


if __name__ == "__main__":
    unittest.main()