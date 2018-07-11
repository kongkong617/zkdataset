import numpy as np
from typing import Dict, List  


def get_resampling( class_info:Dict[str, List], target:Dict[str, List]) -> Dict:
    """
    Argument:
        `class_info`: A Dict[str, List]
            where `str` is the class label, and `List` is a index list of samples
        `target`: A Dict
            represent the target mount of samples
    Return:
        A shuffled tuple
    """
    result = {}
    for k, v in class_info.items():
        lg = len(v)
        if k in target.keys():
            if lg >= target[k]:
                sampling = v[:target[k]]
            else:
                sampling = v[:]
                sampling.extend(np.random.choice(v, target[k]-lg))
        else:
            sampling = v[:]
        
        result.update({k : sampling})

    return result