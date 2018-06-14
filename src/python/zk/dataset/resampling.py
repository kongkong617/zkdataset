import numpy as np
from typing import Dict, List  


def get_resampling( class_info:Dict[str, List], target:int):
    """
    Argument:
        `class_info`: A Dict[str, List]
            where `str` is the class label, and `List` is a index list of samples
        `sampling`: A Integer
            represent the target mount of samples
    Return:
        A shuffled tuple
    """
    result = None
    for _, v in class_info.items():
        v_array = np.array(v)
        sampling = np.random.choice(v_array, target)
        if result is None:
            result = sampling
        else:
            result = np.hstack((result, sampling))

    np.random.shuffle(result)
    return tuple(result)