import unittest
from zk.dataset.resampling import get_resampling 

class TestGetResampling(unittest.TestCase):
    def test_get_resampling(self):
        class_info = {
            '1' : [1, 2],
            '2' : [3, 4, 5, 6]
        }
        target = {
            '1' : 3,
            '2' : 3
        }

        result = get_resampling(class_info, target)

        for k, v in result.items():
            assert len(v) == 3
            if k == '1':
                self.assertListEqual(v[:2], [1, 2])
            else:
                self.assertListEqual(v, [3, 4, 5])
