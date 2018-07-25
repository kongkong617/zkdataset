import os
import asyncio
import logging
import time
import concurrent.futures
import pickle
from collections import OrderedDict
from typing import Dict, List, Tuple
from zk.dataset.mcc2015 import load_label
from zk.dataset.utils import instruction_count


# Path setting
HOME = os.environ['HOME']
MCC2015 = os.path.join(HOME, 'DataSet/kaggle/MCC2015')
ORIASM = os.path.join(MCC2015, 'test/asm')
TRAINLABEL = os.path.join(MCC2015, 'trainLabels.csv')

  
def dir_list(path):
    log = logging.getLogger('master {}'.format(os.getpid())) 

    sample_path = []
    if os.path.exists(path):
        log.info('listdir {}'.format(path))
        asm_samples = os.listdir(path)
    else:
        log.info('{} not exit'.format(path))
        return sample_path

    for item in asm_samples:
        item_path = os.path.join(path, item)
        sample_path.append(item_path)

    return sample_path


def do_make(file_path, label):
    log = logging.getLogger('woker {}'.format(os.getpid()))

    asm_id = os.path.basename(file_path).split('.')[0]
    log.info('parse asm_id = {}'.format(asm_id))
    # asm_label = label[asm_id]
    # log.info('{} is {}'.format(asm_id, asm_label))
    
    ct, lg = instruction_count(file_path)
    log.info('{} : ct={} lg={}'.format(asm_id, ct, lg))
    
    return (ct, lg)


async def dispatch_work(executor, sample_path):
    log = logging.getLogger('master {}'.format(os.getpid()))

    log.info('load label from {}'.format(TRAINLABEL))
    label = load_label(TRAINLABEL)

    log.info('creating executor tasks')
    loop = asyncio.get_event_loop()
    update_tasks = [
        loop.run_in_executor(executor, do_make, sample, label)
        for sample in sample_path
    ]
    log.info('waiting for executor tasks...')

    completed, _ = await asyncio.wait(update_tasks)
    results = [t.result() for t in completed]
    textlen_per_file = {}
    codelen_per_text = {}

    for atuple in results:
        ct, lg = atuple   # parse textlen
        if ct in textlen_per_file:
            textlen_per_file[ct] += 1
        else:
            textlen_per_file[ct] = 1

        for k, v in lg.items(): # parse codelen
            if k in codelen_per_text:
                codelen_per_text[k] += v
            else:
                codelen_per_text[k] = v

    # orderdict
    order_textlen = OrderedDict(sorted(textlen_per_file.items()))
    order_codelen = OrderedDict(sorted(codelen_per_text.items()))

    print(order_textlen)
    print(order_codelen)

    # save
    with open("mcc2015_textlen_per_file", 'wb') as f: 
        pickle.dump(order_textlen, f, -1)
    with open("mcc2015_codelen_per_text", 'wb') as f:
        pickle.dump(order_codelen, f, -1)
    
    log.info('mcc2015 asm count tasks is done!!!')


if __name__ == '__main__':
    strf_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)8s %(levelname)4s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=('mcc2015_asm_counter.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    log = logging.getLogger('').addHandler(console)

    start_time = time.time()
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    logging.info('mcc2015 asm count start %s' % local_time)
    
    # ================= muti-executor start ================ #
    sample_path = dir_list(ORIASM)
    if not sample_path:
        logging.error('mcc2015 asm count stop')        
        exit(0)

    event_loop = asyncio.get_event_loop()
    executor = concurrent.futures.ProcessPoolExecutor(
         max_workers=64,
    )
    try:
        event_loop.run_until_complete(
            dispatch_work(executor, sample_path)
        )
    finally:
        event_loop.close()
    # ================= muti-executor end ================ #

    end_time = time.time()
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info('make mcc2015 data stop at %s' % local_time)
    logging.info('time cost = %s h' % ((end_time-start_time)/3600))