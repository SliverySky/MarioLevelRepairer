import json
from utils.level_process import *
import numpy as np
from root import rootpath

class Identify():
    def __init__(self):
        self.mp = {}
        rule_level = json.load(open(rootpath + "//CNet//data//all_elm_rule.json"))
        for data in rule_level:
            h = data[0]
            for i in range(-2, 3):
                data[0] = h + i
                keyv = tuple(data)
                self.mp[keyv] = 1

    def compare(self, before, after):
        with open(before) as f:
            strlv = f.read()
            before = numpy_level(strlv)
        with open(after) as f:
            strlv = f.read()
            after = numpy_level(strlv)
        U = self.getU(before)
        set1 = self.getWrong(before, pos=U)
        set2 = self.getDif(before, after, pos=U)
        set6 = U - set1
        set4 = set1 & set2
        set3 = set1 - set2
        set5 = set2 - set1
        set3_1 = self.getWrong(after, pos=set3)
        set4_1 = self.getWrong(after, pos=set4)
        set5_1 = self.getWrong(after, pos=set5)
        set6_1 = self.getWrong(after, pos=set6)
        # res[0] += len(set4_1)
        # res[1] += len(set4) - len(set4_1)
        # res[2] += len(set5_1)
        # res[3] += len(set5) - len(set5_1)
        # res[4] += len(set3_1)
        # res[5] += len(set3) - len(set3_1)
        # res[6] += len(set6_1)
        # res[7] += len(set6) - len(set6_1)
        print('W->W:',len(set4_1))
        print('W->T:',len(set4)-len(set4_1))
        print('T->W:', len(set5_1))
        print('T->T:',len(set5)-len(set5_1))
        print('W=W:',len(set3_1))
        print('W=T:',len(set3)-len(set3_1))
        print('T=W:',len(set6_1))
        print('T=T:',len(set6)-len(set6_1))

    def getU(self, lv):
        w, h = len(lv), len(lv[0])
        lv = np.lib.pad(lv, (1, 1), 'constant', constant_values=11)
        res = set()
        for i in range(1, w + 1):
            for j in range(1, h + 1):
                val = (i - 1, lv[i - 1][j - 1], lv[i - 1][j], lv[i - 1][j + 1],
                       lv[i][j - 1], lv[i][j + 1], lv[i + 1][j - 1], lv[i + 1][j], lv[i + 1][j + 1], lv[i][j])
                for k in range(1, 10):
                    if val[k] in [6, 7, 8, 9]:
                        res.add((i - 1, j - 1))
                        break
        return res

    def getWrong(self, lv, pos, ouput=False):
        w, h = len(lv), len(lv[0])
        lv = np.lib.pad(lv, (1, 1), 'constant', constant_values=11)
        res = set()
        for x, y in pos:
            i, j = x + 1, y + 1
            val = (i - 1, lv[i - 1][j - 1], lv[i - 1][j], lv[i - 1][j + 1],
                   lv[i][j - 1], lv[i][j + 1], lv[i + 1][j - 1], lv[i + 1][j], lv[i + 1][j + 1], lv[i][j])
            if val not in self.mp.keys():
                if ouput: print(x, y, val)
                res.add((x, y))
        return res

    def getDif(self, lv1, lv2, pos):
        res = set()
        for i, j in pos:
            if (lv1[i][j] != lv2[i][j]):
                res.add((i, j))
        return res
if __name__ == '__main__':
    idf = Identify()
    idf.compare("result//start.txt", "result//result.txt")

# idf = Identify()
# model_name = ['1']
# import xlwt  # 导入模块
#
# wb = xlwt.Workbook(encoding='ascii')  # 创建新的Excel（新的workbook），建议还是用ascii编码
#
# for name in model_name:
#     ws = wb.add_sheet('size' + name)
#     ws.write(0, 0, label='type\model')
#     ws.write(1, 0, label="W->W")
#     ws.write(2, 0, label="W->T")
#     ws.write(3, 0, label="T->W")
#     ws.write(4, 0, label="T->T")
#     ws.write(5, 0, label="W=W")
#     ws.write(6, 0, label="W=T")
#     ws.write(7, 0, label="T=W")
#     ws.write(8, 0, label="T=T")
#     ws.write(9, 0, label="前后错误比")
#     avg = [0] * 9
#     min_wrong = 100000
#     ans = 0
#     for model_i in range(1, 11):
#         res = [0] * 9
#         for lv_i in range(1, 11):
#             print('lv', lv_i)
#             f1 = "./Experiment1/destroyed/lvl" + str(lv_i) + ".txt"
#             f2 = "./Experiment1/repair/entropy/0.3" + "/lvl" + str(lv_i) + "_net" + str(model_i) + ".txt"
#             idf.compare(f1, f2)
#         ws.write(0, model_i, label="net" + str(model_i))
#         for i in range(8):
#             ws.write(i + 1, model_i, label=str(res[i] / 10))
#             avg[i] += res[i] / 10
#         tmp = (res[0] + res[2] + res[4] + res[6]) / (res[0] + res[1] + res[5] + res[5])
#         ws.write(9, model_i, label=str(tmp))
#         avg[8] += tmp
#     ws.write(0, 11, label="Avg")
#     for i in range(9):
#         ws.write(i + 1, 11, label=str(avg[i] / 10))
# wb.save('repair_table.xls')  # 保存为weng.xls文件




# idf = Identify()
#
# for lv_i in range(1, 11):
#     print('lv', lv_i)
#     res = [0] * 9
#     print("====lv",lv_i,"====")
#     f1 = "./Experiment1/destroyed/lvl" + str(lv_i) + ".txt"
#     f2 = "./Experiment1/repair/with_norepair/lvl" + str(lv_i) + ".txt"
#     idf.compare(f1, f2)
#     for j in range(8):
#         print(res[j])
