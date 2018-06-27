import numpy as np
from typing import Dict, List  


def get_resampling( class_info:Dict[str, List], target:Dict[str, List]):
    """
    Argument:
        `class_info`: A Dict[str, List]
            where `str` is the class label, and `List` is a index list of samples
        `target`: A Dict
            represent the target mount of samples
    Return:
        A shuffled tuple
    """
    result = None
    for k, v in class_info.items():
        v_array = np.array(v)
        if k in target.keys():
            sampling = np.random.choice(v_array, target[k])
        else:
            sampling = v_array
            
        if result is None:
            result = sampling
        else:
            result = np.hstack((result, sampling))

    np.random.shuffle(result)
    return tuple(result)