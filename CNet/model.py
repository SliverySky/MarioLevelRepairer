import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json


class CNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2, n_output):
        super(CNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x


if __name__ == '__main__':
    rule_level = json.load(open("./data/rule.json"))
    batch_size = 1
    total = 4000
    USEGPU = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    gpus = [0]
    cnet_num = 1
    for t in range(cnet_num):
        net = CNet(12 * 8 + 1, 200, 100, 12)
        # net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

        # loss_func = torch.nn.MSELoss()
        loss_func = torch.nn.CrossEntropyLoss()
        input = torch.zeros(batch_size, 97, dtype=torch.float32)
        target = torch.zeros(batch_size, 12, dtype=torch.float32)
        input = input.to(device)
        target = target.to(device)
        data1 = []
        data2 = []
        for i in range(len(rule_level)):
            val1 = [0.0] * 97
            # val2 = [0.0] * 12
            val1[0] = rule_level[i][0]
            for k in range(1, 9):
                val1[k * 12 - 11 + rule_level[i][k]] = 1
            # val2[rule_level[i][9]] = 1
            val2 = rule_level[i][9]
            data1.append(val1)
            data2.append(val2)
        data1 = np.array(data1)
        data2 = np.array(data2)
        for i in range(total):
            perm = torch.randperm(len(data1))
            data1 = data1[perm]
            data2 = data2[perm]
            sum = 0
            for j in range(len(rule_level) // batch_size):
                input = Variable(torch.tensor(data1[batch_size * j:batch_size * (j + 1)]).float(), requires_grad=True)
                target = Variable(torch.LongTensor(data2[batch_size * j:batch_size * (j + 1)]))
                optimizer.zero_grad()
                # input = F.softmax(net(input),dim=1)
                input = net(input)
                loss = loss_func(input, target)
                loss.backward()
                optimizer.step()
                if USEGPU:
                    sum += loss.cpu().detach().numpy()
                else:
                    sum += loss.detach().numpy()
            print("\rNet", t + 1, "(size=", batch_size, ")iter=", i, "/", total, end='')
            print("     loss=", sum)
        torch.save(net, "dict.pkl")
        # for i in range(100):
        #     shuffle(rule_level)
        #     for j0 in range(len(rule_level)//batch_size):
        #         first_flag=True
        #         label =[]
        #         prediction=[]
        #         for l in range(batch_size):
        #             j=j0*batch_size+l
        #             x = np.zeros(97)
        #             x[0]=rule_level[j][0]
        #             for k in range(1,9):
        #                 x[k*12-11+rule_level[j][k]]=1
        #             newX=x
        #             newX = Variable(torch.tensor(newX).float())
        #             if(first_flag):
        #                 prediction = net(newX).unsqueeze(0)
        #                 first_flag=False
        #             else:
        #                 prediction = torch.cat((prediction,net(newX).unsqueeze(0)),dim=0)
        #             label.append(rule_level[j][9])
        #         label = Variable(torch.tensor(label))
        #         prediction=F.softmax(prediction,dim=1)
        #         optimizer.zero_grad()
        #         loss = loss_func(prediction,label)
        #         loss.backward()
        #         optimizer.step()
        #     print("iter=",i)
