import torch
import json
import torch.nn.functional as F
from CNet.model import CNet     # Don't delete it

net = torch.load("dict.pkl").to("cpu")


def cal(name):
    cnt, total = [0] * 11, [0] * 11
    rule = json.load(open(name))
    threshold = 0.01
    for i in rule:
        data = i[0:9]
        y = i[9]
        x = torch.zeros(97)
        x[0] = data[0]
        for i in range(1, 9):
            x[i * 12 - 11 + data[i]] = 1
        if F.softmax(net(x), dim=0)[y] < threshold:
            cnt[y] += 1
        total[y] += 1
    return cnt, total

def test_legal(name, change_n):
    path = 'data//' + name + ("" if change_n == 0 else ("_F" + str(change_n))) + ".json"
    cnt, total = cal(path)
    cnt_sum, total_sum = 0, 0
    for i in range(len(cnt)):
        print('Type', i, " Elimated=", cnt[i], "/", total[i],
              ' Rate=%.1f%%' % (100 * cnt[i] / total[i]) if total[i] else 0)
        cnt_sum += cnt[i]
        total_sum += total[i]
    print('Sum ', " Elimate=", cnt_sum, "/", total_sum, ' Rate=', cnt_sum / total_sum)

def test_illegal(name, change_n):
    path = 'data//' + name + ("" if change_n == 0 else ("_F" + str)) + ".json"
    cnt, total = cal(path)
    cnt_sum, total_sum = 0, 0
    for i in range(len(cnt)):
        print('Type', i, " Detect=", cnt[i], "/", total[i], ' Rate=%.1f' % (100 * cnt[i] / total[i]) if total[i] else 0)
        cnt_sum += cnt[i]
        total_sum += total[i]
    print('Sum ', " Detect=", cnt_sum, "/", total_sum, ' Rate=', cnt_sum / total_sum)


if __name__ == '__main__':
    print(
        "Experiment 1. Results of identifying legal tiles. The number of legal tiles which are wrongly eliminated by CNet.")
    for i in range(4):
        print("True surrounding info" if i == 0 else "Fake surrounding info with " + str(
            i) + "randomly" + "changed element")
        test_legal('legal_rule', i)
    print(
        "Experiment 2. Results of detecting illegal tiles. The number of illegal tiles which are detected correctly by CNet.")
    for i in range(4):
        print("True surrounding info" if i == 0 else "Fake surrounding info with " + str(
            i) + "randomly" + "changed element")
        test_legal('illegal_rule', i)
