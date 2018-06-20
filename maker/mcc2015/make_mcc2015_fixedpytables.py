import os
from zk.dataset.mcc2015 import DirDataGenerator, MccFixedPytablesMaker

# Path setting
HOME = os.environ['HOME']
MCC2015 = os.path.join(HOME, 'DataSet/kaggle/MCC2015')
DTRAIN = os.path.join(MCC2015, 'dtrain')

TBPATH = os.path.join(MCC2015, "mcc2015_fixed.h5")
NB_CLASS = 9
TOTAL_SAMPLE = 10860
SHAPE = (50, 1024, 8)


def main():
    data_gen = DirDataGenerator(DTRAIN)
    tbmaker = MccFixedPytablesMaker(TBPATH, NB_CLASS, SHAPE, data_gen)
    tbmaker.make()
    print(tbmaker.capacity)
    with open("mcc2015-fixed-h5.log", "w") as f:
        for k, v in tbmaker.capacity.items():
            line = line = "{}  : {}".format(k, v)
            f.writelines(line + os.linesep)

if __name__ == "__main__":
    main()