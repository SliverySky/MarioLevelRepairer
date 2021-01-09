from utils.level_process import *
if __name__ == '__main__':
    num = 1
    lv_str = ''
    with open(rootpath + "//LevelGenerator//RandomDestroyed//lvl" + str(1) + ".txt") as f:
        for i in f.readlines():
            lv_str += i
    lv = numpyLevel(lv_str)
    print(lv_str)
    for i in range(num):
        new_lv = random_destroy(lv)
        print(arr_to_str(new_lv))
