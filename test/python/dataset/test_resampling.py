import unittest
from zk.dataset.resampling import get_resampling 

class TestGetResampling(unittest.TestCase):
    def test_get_resampling(self):
        class_info = {
            '1' : [1, 2],
            '2' : [3, 4, 5, 6]
        }
        target = 3

        result = get_resampling(class_info, target)

        assert len(result) == 6
        class_1 = result.count(1) + result.count(2)
        assert class_1 == 3