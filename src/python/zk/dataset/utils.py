import re
import codecs
from typing import Dict, List, Tuple



def instruction_length(aline:str):
    oplist = []
    p = re.compile(r'^[0-9A-F]{1,2}[0-9A-F\+]$')
    for code in aline.split()[1:]:
        if len(code) not in [2, 3]:
            continue
        s_obj = re.search(p, code)
        if s_obj: 
            oplist.extend(s_obj.group()[:2])
    
    lg = len(oplist) * 4
    if lg >= 120:    # valid max length is 120 bits
        return 120
    else:

        return lg


def instruction_count(fname) -> (int, Dict):
    _count = 0
    _length = {}
    with codecs.open(fname, mode="r", encoding='utf-8', errors='ignore') as f:
        data = f.read()

        for line in data.splitlines():
            if line[0:5] == '.idata':    # .text is over
                break
            if line[0:5] == '.text':
                lg = instruction_length(line)
                if lg == 0:
                    continue
                else:
                    if lg in _length:
                        _length[lg] = _length[lg] + 1
                    else:
                        _length[lg] = 1
                        
                    _count += 1
    
    return _count, _length