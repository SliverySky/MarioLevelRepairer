import json
import numpy as np
import random
from root import rootpath
import os

map_dic = {'X': 0, 'S': 1, '-': 2, '?': 3, 'Q': 4, 'E': 5, '<': 6, '>': 7, '[': 8, ']': 9, 'o': 10}
phareser = ['X', 'S', '-', '?', 'Q', 'E', '<', '>', '[', ']', 'o']


# def getContinueData(file_name):
#     file = open(file_name)
#     data = file.readlines()
#     height = len(data)
#     width = len(data[0]) - 1  # '\n' is not included
#     whole_level = np.empty((height, width), dtype=int, order='C')
#     for i in range(height):
#         for j in range(width):
#             whole_level[i][j] = map_dic[data[i][j]]
#     level_list = []
#     clip_len = 28  # set width of clipped map
#     for i in range(width // clip_len):
#         clipped_level = whole_level[0:height, i * clip_len:(i + 1) * clip_len]
#         clipped_level = extendLevel(clipped_level, 1)
#         level_list.append(clipped_level.tolist())
#     return level_list


def getRuleData():
    names = []
    for root, dirs, files in os.walk(rootpath + '\\LevelText\\MarioBrother2'):
        for fl in files:
            names.append(rootpath + '\\LevelText\\MarioBrother2\\' + fl)
    rule_set = set()
    for file_name in names:
        file = open(file_name)
        data = file.readlines()
        height = len(data)
        width = len(data[0]) - 1  # '\n' is not included
        for i in range(height):
            for j in range(width):
                flag = False
                whole_level = []
                whole_level.append(i);
                for i1 in range(-1, 2):
                    for j1 in range(-1, 2):
                        ni = i + i1
                        nj = j + j1
                        if (ni == i and nj == j): continue
                        if (ni < 0 or nj < 0 or ni >= height or nj >= width):
                            whole_level.append(11)
                        else:
                            whole_level.append(map_dic[data[ni][nj]])
                            if (data[ni][nj] == '<' or data[ni][nj] == '>' or data[ni][nj] == '[' or data[ni][
                                nj] == ']'):
                                flag = True
                whole_level.append(map_dic[data[i][j]])
                if (flag): rule_set.add(tuple(whole_level))
    with open(rootpath+"/CNet/data/legal_rule.json", "w") as f:
        json.dump(list(rule_set), f)


def getAllElmRuleData():
    names = []
    for root, dirs, files in os.walk(rootpath + '\\Level\\MarioBrother2'):
        for fl in files:
            names.append(rootpath + '\\Level\\MarioBrother2\\' + fl)
    rule_set = set()
    for file_name in names:
        file = open(file_name)
        data = file.readlines()
        height = len(data)
        width = len(data[0]) - 1  # '\n' is not included
        for i in range(height):
            for j in range(width):
                flag = False
                whole_level = []
                whole_level.append(i);
                for i1 in range(-1, 2):
                    for j1 in range(-1, 2):
                        ni = i + i1
                        nj = j + j1
                        if (ni == i and nj == j): continue
                        if (ni < 0 or nj < 0 or ni >= height or nj >= width):
                            whole_level.append(11)
                        else:
                            whole_level.append(map_dic[data[ni][nj]])
                whole_level.append(map_dic[data[i][j]])
                rule_set.add(tuple(whole_level))
    with open('all_elm_rule.json', "w") as f:
        json.dump(list(rule_set), f)


def convert(ch):
    return map_dic[ch]


# number with string
def arr_to_str(level):
    height = len(level)
    width = len(level[0])
    str = ''
    for i in range(height):
        for j in range(width):
            str += phareser[level[i][j]]
        if i < height - 1:
            str += '\n'
    return str


def numpy_level(str):
    data = str.split('\n')

    height = len(data)
    if len(data[height - 1]) == 0: height -= 1
    width = len(data[0])  # '\n' is not included
    whole_level = np.empty((height, width), dtype=int, order='C')
    for i in range(height):
        for j in range(width):
            whole_level[i][j] = map_dic[data[i][j]]
    return whole_level
#
#
# def connectLevels(levelList):
#     newLevel = [[] for i in range(len(levelList[0]))]
#     for level in levelList:
#         for j in range(len(level)):
#             newLevel[j].extend(level[j])
#     return newLevel
#
#
# # add bottom and top a same line
# def addLine(level0):
#     level = level0.copy()
#     level.insert(0, level[0].copy())
#     level.append(level[len(level) - 1].copy())
#     return level
def random_destroy(level, p=0.2):

    new_level = level.copy()
    h, w = level.shape
    for i in range(h):
        for j in range(w):
            window = new_level[i-1:i+1, j-1:j+1]
            window = window.reshape(-1)
            flag = False
            for e in window:
                if 6 <= e <= 9:
                    flag = True
                    break
            if flag and random.random() < p:
                prev = new_level[i][j]
                while new_level[i][j] == prev:
                    new_level[i][j] = random.randrange(11)
    return new_level
def little_level(level, size):
    height = len(level)
    width = len(level[0])
    litte_level = np.empty((height // size, width // size), dtype=int, order='C')
    cnt = [0] * len(map_dic.keys())
    for i in range(height // size):
        for j in range(width // size):
            for k in range(len(map_dic.keys())):
                cnt[k] = 0
            for k in range(size):
                for l in range(size):
                    cnt[level[i * size + k][j * size + l]] += 1
            litte_level[i][j] = random.sample(list(np.where(cnt == np.max(cnt))[0]), 1)[0]
    return litte_level
def addLine(lv):
    n = len(lv)
    return np.concatenate([lv[0:1], lv[0:n], lv[n-1:n]], axis=0)


def calculate_broken_pipes(data):
    rule_file = json.load(open(rootpath + '//CNet//data//legal_rule.json'))
    rule = set()
    for e in rule_file:
        rule.add(tuple(e))
    height = len(data)
    width = len(data[0])
    cnt = 0
    for i in range(height):
        for j in range(width):
            flag = False
            info = []
            info.append(i)
            for i1 in range(-1, 2):
                for j1 in range(-1, 2):
                    ni = i + i1
                    nj = j + j1
                    if ni == i and nj == j:
                        continue
                    if ni < 0 or nj < 0 or ni >= height or nj >= width:
                        info.append(11)
                    else:
                        info.append(data[ni][nj])
                        if data[ni][nj] >=6 and data[ni][nj] <= 9:
                            flag = True
            info.append(data[i][j])
            info = np.array(info)
            if flag and tuple(info) not in rule:
                cnt += 1
    return cnt