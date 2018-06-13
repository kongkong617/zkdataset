import os
from zk.dataset.mcc2015 import DirDataGenerator, MccPytablesMaker

# Path setting
HOME = os.environ['HOME']
MCC2015 = os.path.join(HOME, 'DataSet/kaggle/MCC2015')
DTRAIN = os.path.join(MCC2015, 'dtrain')

TBPATH = os.path.join(MCC2015, "mcc2015.h5")
NB_CLASS = 9
TOTAL_SAMPLE = 10860


def main():
    data_gen = DirDataGenerator(DTRAIN)
    tbmaker = MccPytablesMaker(TBPATH, NB_CLASS, data_gen)
    tbmaker.make() 
    assert tbmaker.capacity == TOTAL_SAMPLE
    tbmaker.close()


if __name__ == "__main__":
    main()