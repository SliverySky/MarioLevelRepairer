import torch
import torch.nn.functional as F
import copy
import math
import random
import numpy
import matplotlib.pyplot as plt
from utils.level_process import *
from utils.visualization import *
from CNet.model import CNet
from root import rootpath

Threshold=0.05
origin=None
net=None
P_M0=0.8
P_M1=0 #1/len(S)
RRT_M=4
Lamda=20
Iteration=50
RepairRatio=0.3
S=[]
hash_map={}
repair_set={}
picture_name=''
xv=[]
yv1=[]
yv2=[]
yv3=[]

score=[]
bestScore=[]
class ruleNet(torch.nn.Module):
    def __init__(self,n_feature, n_hidden,n_hidden2, n_output):
        super(ruleNet,self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x
def crossOver(ind1,ind2):
    indvd1=copy.deepcopy(ind1)
    indvd2=copy.deepcopy(ind2)
    for i in S:
        if(random.random()<0.5):
            tmp = indvd1[i]
            indvd1[i] = indvd2[i]
            indvd2[i] = tmp
    return indvd1,indvd2
def repairTile(ind,):
    pos=[]
    for item in S:
        flag,pro_tile = getProTile(ind,item[0],item[1])
        if flag:
            if(len(pro_tile)>0 and ind[item] not in pro_tile):
                pos.append(item)
    random.shuffle(pos)
    for v in range(len(pos)):
        if random.random() < RepairRatio:
            item = pos[v]
            flag, pro_tile = getProTile(ind, item[0], item[1])
            if flag:
                if (len(pro_tile) > 0 and ind[item] not in pro_tile):
                    ind[item] = pro_tile[int(random.random()*len(pro_tile))]
def mutation(ind):
    for item in S:
        if(random.random()<P_M1):
            if(random.random()<0.5):
                flag,pro_tile = getProTile(ind,item[0],item[1])
                if flag:
                    ind[item] = pro_tile[int(random.random()*len(pro_tile))]
            else:
                ind[item] = origin[item[0]][item[1]]
def getStep(ind):
    cnt=0
    for i,j in S:
        if origin[i][j]!=ind[(i,j)]:
            cnt+=1
    return cnt
def select(pop,children):
    big=[]
    for ind in pop:
        ind['RRT']=0
        big.append(ind)
    for ind in children:
        ind['RRT']=0
        big.append(ind)
    for i in range(len(big)):
        cnt=0
        while cnt<RRT_M:
            tmp = int(random.random()*len(big))
            if tmp!=i:
                if big[i]['fit']<big[tmp]['fit']:
                    big[i]['RRT']-=1
                else:
                    big[tmp]['RRT']-=1
                cnt+=1
    big.sort(key=lambda x:x['RRT'])
    big=big[:Lamda]
    return big
def inMap(i,j):
    if i>=0 and i<len(origin) and j>=0 and j<len(origin[0]):
        return True
    else:
        return False
def updateProbility(pop):
    # sty sort
    pop.sort(key=lambda  x:x['fit'])
    sum=(1+len(pop))*len(pop)/2
    for i in range(len(pop)):
        pop[i]['p']=(i+1)/sum
def randomChooseInd(pop):
    x=random.random()
    cnt=0
    for i in range(len(pop)):
        cnt += pop[i]['p']
        if cnt>x:
            return i
    print('not find:',cnt,x)
    return len(pop)-1
def evolution(isfigure:bool=False, isrepair:bool=False):
    figure_index=[0,1,2,4,8,15,25,40,60,90]
    global S
    S=[]
    initial()
    pop=[]
    pop=initpop()
    best = pop[0]
    start={}
    for i,j in S:
        start[(i,j)]=origin[i][j]
    level, S1, T1 = getMarkSet(start)
    if(isfigure):
        saveLevelAsImage(level,"Experiment0//"+"start")
        saveAndMark(level, "Experiment0//"+"start(Remark)", T1, S1)
    for index in range(Iteration):
        if index in figure_index:
            if (isfigure):
                level,S1,T1 = getMarkSet(best)
                saveAndMark(level,"Experiment0//"+"iteration"+str(index),T1,S1)
        for ind in pop:
            updateFitness(ind)
        for i in range(Lamda):
            for j in score[index][i].keys():
                #print(j)
                score[index][i][j]+=pop[i][j]
                #print(index,j,pop[i][j])
        updateProbility(pop)
        children=[]
        while len(children)<Lamda:
            x,y=None,None
            while True:
                x,y=randomChooseInd(pop),randomChooseInd(pop)
                if(x!=y):break
            x1,y1=crossOver(pop[x],pop[y])
            children.append(x1)
            children.append(y1)

        #update score

        for ind in children:
            if(random.random()<P_M0):
                mutation(ind)
        if isrepair:
            for ind in children:
                repairTile(ind)
        for ind in children:
            updateFitness(ind)
        pop = select(pop,children)
        for ind in pop:
            if ind['fit']<best['fit']:
                best=copy.deepcopy(ind)
        xv.append(index)
        avg=0
        for ind in pop:
            avg+=ind['fit']
        print("iter=", index, "best_fit=", best['fit'],'avg=',avg/Lamda)
    '''for i in range(Lamda):
        for j in score[Iteration].keys():
            score[Iteration][j] += pop[i][j]'''
    level, S1, T1 = getMarkSet(best)
    if(isfigure):
        saveLevelAsImage(level,"Experiment0//"+"result")
        saveAndMark(level, "Experiment0//"+"result(Remark)", T1, S1)
    return best
def useNet():
    pass
def getProTile(ind,i,j):
    flag = False
    condition = []
    condition.append(i);
    height = len(origin)
    width = len(origin[0])
    for offset in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]:
        ni = i + offset[0]
        nj = j + offset[1]
        if (ni < 0 or nj < 0 or ni >= height or nj >= width):
            condition.append(11)
        else:
            if (ni, nj) in ind.keys():
                tmp = ind[(ni, nj)]
            else:
                tmp = origin[ni][nj]
            if (tmp >= 6  and tmp<=9):
                flag = True
            if (ni == i and nj == j):
                continue
            condition.append(tmp)
    if (flag):
        if(tuple(condition) in hash_map.keys()):
            pro_tile = hash_map[tuple(condition)]
        else:
            if (i,j) in ind.keys():
                y = ind[(i,j)]
            else:
                y=origin[i][j]
            x = torch.zeros(97)
            x[0] = condition[0]
            for i1 in range(1, 9):
                x[i1 * 12 - 11 + condition[i1]] = 1
            pro = F.softmax(net(x), dim=0)
            if(tuple(condition)==(13, 2, 2, 8, 2, 8, 2, 2, 2)):
                print(pro)
            pro_tile = []
            pro_num = []
            for i1 in range(11):
                if (pro[i1] >= Threshold):
                    pro_tile.append(i1)
                    pro_num.append(pro[i1])
            # Z = zip(pro_num,pro_tile)
            # sorted(Z)
            hash_map[tuple(condition)]=pro_tile
        #if (i == 13 and j == 75): print(condition,pro_tile)
        return True,pro_tile
    else:
        return False,[]
def fitness_fuction(pro_sum,error_sum,step):
    return pro_sum+5*error_sum+3*step
def initial():
    global P_M1
    height = len(origin)
    width = len(origin[0])
    for i in range(height):
        for j in range(width):
            flag,pro_tile=getProTile({},i,j)
            if flag:
                if len(pro_tile)>1 or origin[i][j] not in pro_tile:
                    S.append((i,j))
                    #print(i,j)
    print("#S=",len(S))
    P_M1=1/len(S)
def initpop():
    pop=[]
    pos=[]
    for i in range(Lamda):
        ind={}
        for item in S:
            ind[item]=origin[item[0]][item[1]]
        pop.append(ind)
    for item in S:
        flag, pro_tile = getProTile({}, item[0], item[1])
        if flag and origin[item[0]][item[1]] in pro_tile:
            pos.append(item)
    for i in range(Lamda):
        random.shuffle(pos)
        for item in pos:
            flag, pro_tile = getProTile(pop[i], item[0], item[1])
            if flag and pop[i][item] in pro_tile:
                pop[i][item] = pro_tile[int(random.random() * len(pro_tile))]
        repairTile(pop[i])
    return pop
def updateFitness(ind):
    pro_sum=0
    error_sum=0

    for i,j in S:
        flag, pro_tile = getProTile(ind, i, j)
        if flag:
            if (ind[(i,j)] in pro_tile) and len(pro_tile) > 1:
                pro_sum += len(pro_tile)
            elif (ind[(i,j)] not in pro_tile):
                error_sum += 1
                pro_sum += len(pro_tile)

    ind['wrong']=error_sum
    ind['value']=pro_sum
    ind['replace']=getStep(ind)
    ind['fit']=fitness_fuction(pro_sum,error_sum,ind['replace'])
    #print(ind['wrong'],ind['value'],ind['replace'])
def getMarkSet(ind):
    S1=[]
    T1=[]
    level = copy.deepcopy(origin)
    for i,j in S:
        level[i][j]=ind[(i,j)]
    for i,j in S:
        flag, pro_tile = getProTile(ind, i, j)
        if flag:
            if (ind[(i,j)] in pro_tile) and len(pro_tile) > 1:
                S1.append((i,j))
            elif (ind[(i,j)] not in pro_tile):
                T1.append((i,j))
    return level,S1,T1
def repair(best):
    level=copy.deepcopy(origin)
    for i,j in S:
        level[i][j]=best[(i,j)]
    return level
def drawLineGraph(total):
    x,y1,y2,y3,y4=[],[],[],[],[]
    for i in range(Iteration):
        x.append(i)
        res={'fit':0, 'wrong':0, 'replace':0, 'value':0}
        for j in range(Lamda):
            for k in score[i][j].keys():
                res[k]+=score[i][j][k]
        y1.append(res['fit']/Lamda/total)
        y2.append(5*res['wrong']/Lamda/total)
        y3.append(3*res['replace']/Lamda/total)
        y4.append(res['value']/Lamda/total)
    plt.plot(x,y1,color='r',linestyle='-',label='Fitness')
    plt.plot(x,y2,'b-',label='5*#Wrong')
    plt.plot(x,y3,'g-',label='3*#Repalce')
    plt.plot(x,y4,'y-',label='UV')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
    #            ncol=3, mode="expand", borderaxespad=0.)
    plt.xlabel("Iteration")
    plt.ylabel("value")
    plt.savefig("plt.png")
def drawPointGraph(total):
    x,y=[],[]
    for i in range(Iteration):
        for j in range(Lamda):
            x.append(i)
            y.append(score[i][j]['fit']/total)
    plt.scatter(x,y,s=10,marker = 'x',label='Fitness')
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.savefig("plt.png")
def drawPic2(total,isBest=False):
    x,y1,y2,y3,y4=[],[],[],[],[]
    for i in range(Iteration):
        x.append(i)
        if(isBest):
            y1.append(bestScore[i]['fit']/total)
            y2.append(5*bestScore[i]['wrong']/total)
            y3.append(3*bestScore[i]['replace']/total)
            y4.append(bestScore[i]['value']/total)
        else:
            y1.append(score[i]['fit']/Lamda/total)
            y2.append(5*score[i]['wrong']/Lamda/total)
            y3.append(3*score[i]['replace']/Lamda/total)
            y4.append(score[i]['value']/Lamda/total)
    plt.plot(x,y1,color='r',linestyle='--',label='Fitness(R)')
    plt.plot(x,y2,'b--',label='5*#Wrong(R)')
    plt.plot(x,y3,'g--',label='3*#Repalce(R)')
    plt.plot(x,y4,'y--',label='UV(R)')
    # plt.legend( loc=9,
    #            ncol=4)
    plt.xlabel("Iteration")
    plt.ylabel("value")

Total=30
net=torch.load(rootpath+"\CNet\dict.pkl")
net.eval()
#type_name="./C_net/size1"+"/ruleNet_"+str(1)+"_sz1"+"_t2000.pkl"
#net = torch.load(type_name)
#for lvl_i in range(1,11):
score = []
for i in range(Iteration):
    one = []
    for j in range(Lamda):
        one.append({"fit": 0, "value": 0, "replace": 0, "wrong": 0})
    score.append(one)
with open(rootpath + "//LevelGenerator//RandomDestroyed//lvl"+str(1)+".txt") as f:
#with open("./txt/repair_mario_damage1.txt") as f:
    data = f.readlines()
    lv_str = ''
    for i in data:
        lv_str+=i
print(lv_str)
print()
for i in range(Total):
    whole_level=numpyLevel(lv_str)
    origin = whole_level
    best = evolution(isfigure=False ,isrepair=True)
    repair_lv = repair(best)
    #saveLevelAsImage(repair_lv,"lvl_"+str(lvl_i))
    print(arr_to_str(repair_lv))
#drawPointGraph(Total)
drawLineGraph(Total)
plt.show()


# above is for two graph
# score = []
# bestScore = []
# for i in range(Iteration):
#     score.append({"fit": 0, "value": 0, "replace": 0, "wrong": 0})
#     bestScore.append({"fit": 0, "value": 0, "replace": 0, "wrong": 0})
# for k in range(Total):
#     with open("./repair_mario_damage2.txt") as f:
#         data = f.readlines()
#         lv_str=''
#         for i in data:
#             lv_str+=i
#     print(lv_str)
#     print()
#     whole_level=numpyLevel(lv_str)
#     origin = whole_level
#     best = evolution(isrepair=False)
#     repair_lv = repair(best)
#     saveLevelAsImage(repair_lv, "figure\\v1")
#
# drawPic(Total,isBest=True)
# score = []
# bestScore = []
# for i in range(Iteration):
#     score.append({"fit": 0, "value": 0, "replace": 0, "wrong": 0})
#     bestScore.append({"fit": 0, "value": 0, "replace": 0, "wrong": 0})
# for k in range(Total):
#     with open("./repair_mario_damage2.txt") as f:
#         data = f.readlines()
#         lv_str = ''
#         for i in data:
#             lv_str += i
#     print(lv_str)
#     print()
#     whole_level = numpyLevel(lv_str)
#     origin = whole_level
#     best = evolution(isrepair=True)
#     repair_lv = repair(best)
#     saveLevelAsImage(repair_lv, "figure\\v1_r")
#
# drawPic2(Total,isBest=True)
# plt.show()
#     saveLevelAsImage(repair_lv,"fig3\\final")
#     with open("./Experiment1/repair/size"+name+"/lvl"+str(lv_i)+"_net"+str(model_i)+".txt","w") as f:
#         f.write(String(repair_lv))
#     print(String(repair_lv))











# import level
# model_name=['1']
# for name in model_name:
#     for model_i in range(1,11):
#         print("start:",name,model_i)
#         #net=torch.load("ruleNet_new.pkl")
#         type_name="./C_net/Entropy/size"+name+"/ruleNet_"+str(model_i)+"_sz"+name+"_t4000.pkl"
#         net = torch.load(type_name)
#         print(type_name)
#         for lv_i in range(1,11):
#             score=[]
#             for i in range(Iteration):
#                 one = []
#                 for j in range(Lamda):
#                     one.append({"fit": 0, "value": 0, "replace": 0, "wrong": 0})
#                 score.append(one)
#             with open("./Experiment1/destroyed/lvl"+str(lv_i)+".txt") as f:
#                 data = f.readlines()
#                 lv_str=''
#                 for i in data:
#                     lv_str+=i
#             print(lv_str)
#             print()
#             whole_level=numpyLevel(lv_str)
#             picture_name="here"
#             origin = whole_level
#             best = evolution(isrepair=True)
#             repair_lv = repair(best)
#             #saveLevelAsImage(repair_lv,"fig3\\final")
#             with open("./Experiment1/repair/entropy/0.3"+"/lvl"+str(lv_i)+"_net"+str(model_i)+".txt","w") as f:
#                 f.write(String(repair_lv))
#             print(String(repair_lv))

