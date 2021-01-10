from GA.repair import *


if __name__ == '__main__':
    render_path = rootpath + "//GA//real_time_renderer.py"
    net_name = rootpath + "//CNet//dict.pkl"
    lv_name = rootpath + "//LevelGenerator//RandomDestroyed//lv" + str(0) + ".txt"
    result_path = rootpath + "//GA//result"
    lv = ""
    for e in open(lv_name).readlines():
        lv = lv + e
    lv = numpy_level(lv)
    GA(net_name, lv_name, result_path, isfigure=True, isrepair=True)
