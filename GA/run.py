import sys,os
sys.path.append(os.path.dirname(sys.path[0]))
from GA.repair import *


if __name__ == '__main__':
    net_name = rootpath + "//CNet//dict.pkl"
    lv_name = rootpath + "//LevelGenerator//RandomDestroyed//lv" + str(0) + ".txt"
    result_path = rootpath + "//GA//result"
    lv = ""
    print('repair lv:',lv_name)
    print('used CNet:',net_name)
    print('saved path:',result_path)
    for e in open(lv_name).readlines():
        lv = lv + e
    lv = numpy_level(lv)
    GA(net_name, lv_name, result_path, isfigure=True, isrepair=True)
