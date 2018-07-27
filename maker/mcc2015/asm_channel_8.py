import os
import asyncio
import logging
import time
import concurrent.futures
import numpy as np
from typing import Dict, List, Tuple

# Path setting
HOME = os.environ['HOME']
MCC2015 = os.path.join(HOME, 'DataSet/kaggle/MCC2015')
ORIASM = os.path.join(MCC2015, 'asm_channel_1_3200')
DTRAIN = os.path.join(MCC2015, 'asm_channel_8_3200')

# make setting
NEWSHAPE = (400, 64, 8)
ORDER = (3, 2, 1)


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
            yield self.asmop_np[i, :, 0]


def make_path(file_path):
    log = logging.getLogger('master {}'.format(os.getpid())) 
    if not os.path.exists(file_path):
        try:
            os.makedirs(file_path)
        except Exception as e:
            log.error(e)
            return None
    
    return file_path 

def dirlist(path):
    log = logging.getLogger('master {}'.format(os.getpid())) 

    sample_path = []
    for dirpath, _, filenames in os.walk(path):
        if filenames:
            for d in filenames:
                sample_path.append(os.path.join(dirpath, d))

    return sample_path

def _resize(ori_np, newshape):
    log = logging.getLogger('woker {}'.format(os.getpid()))
    (dim1, dim2, dim3) = newshape
    padding = np.zeros(newshape)

    asmgen = AsmopGenerator(ori_np)
    log.info("start _resize...")
    for i_row in range(dim1):
        for i_channel in range(dim3):
            try:
                padding[i_row, : ,i_channel] = next(asmgen)
            except StopIteration as e:
                log.error(e)

    return padding

def do_make(file_path):
    log = logging.getLogger('woker {}'.format(os.getpid()))

    asm_id = os.path.basename(file_path).split('.')[0]
    log.info('parse asm_id = {}'.format(asm_id))
    asm_label = os.path.basename(os.path.dirname(file_path))
    log.info('{} is {}'.format(asm_id, asm_label))
    
    ori_np = np.load(file_path)
    log.info('{} ori shape is {}'.format(asm_id, ori_np.shape))
    resize_np = _resize(ori_np, NEWSHAPE)
    log.info('{} resized shape is {}'.format(asm_id, resize_np.shape))

    label_path = os.path.join(DTRAIN, asm_label)
    if make_path(label_path):
        file_path = os.path.join(label_path, asm_id+".npy")
        np.save(file_path, resize_np)
        log.info('save {}'.format(file_path))
    else:
        return False

    return True


async def dispatch_work(executor, sample_path):
    log = logging.getLogger('master {}'.format(os.getpid()))

    log.info('creating executor tasks')
    loop = asyncio.get_event_loop()
    update_tasks = [
        loop.run_in_executor(executor, do_make, sample)
        for sample in sample_path
    ]
    log.info('waiting for executor tasks...')

    completed, _ = await asyncio.wait(update_tasks)
    results = [t.result() for t in completed]
    sum_make = 0
    for item in results:
        if item:
            sum_make += 1

    log.info('deal sample = {}'.format(sum_make))

    log.info('make mcc2015 tasks is done!!!')


if __name__ == '__main__':
    strf_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)8s %(levelname)4s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=('mcc2015_maker_channel_8_3200.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    log = logging.getLogger('').addHandler(console)

    start_time = time.time()
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    logging.info('make mcc2015 dataset at %s' % local_time)
    
    # ================= muti-executor start ================ #
    # sample_path = dirlist(ORIASM)
    # if not sample_path:
    #     logging.error('make mcc2015 dataset stop')        
    #     exit(0)

    # event_loop = asyncio.get_event_loop()
    # executor = concurrent.futures.ProcessPoolExecutor(
    #      max_workers=64,
    # )
    # try:
    #     event_loop.run_until_complete(
    #         dispatch_work(executor, sample_path)
    #     )
    # finally:
    #     event_loop.close()
    # ================= muti-executor end ================ #

    # ================= single-executor start ============ #
    sample_path = dirlist(ORIASM)
    for i in sample_path:
        do_make(i)

    end_time = time.time()
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info('make mcc2015 data stop at %s' % local_time)
    logging.info('time cost = %s h' % ((end_time-start_time)/3600))