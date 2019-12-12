import numpy as np
import networkx as nx
import random as rd
import matplotlib.pyplot as plt
import math
from functools import reduce
import datetime


def func_and(a,b):
    return a and b



class net():
    def __init__(self,points_num=10):
        self.points_num=points_num

    def edges(self):
        pass

    def length(self,way):
        return reduce(lambda x,y:x+y,[self.edges()[way[i]][way[i+1]] for i in range(self.points_num-1)])


class map_2d(net):
    def __init__(self, x_range=100, y_range=100, points_num=10):
        net.__init__(self,points_num)
        self.x_range=x_range
        self.y_range=y_range
        self.points = [[rd.uniform(0,x_range),rd.uniform(0,y_range)] for i in range(points_num)]

    def show(self):
        plt.plot(self.points,'ro')
        plt.show()

    def edges(self):
        return np.array([[math.sqrt((pi[0]-qi[0])**2+(pi[1]-qi[1])**2) for pi in self.points] for qi in self.points])


class Solver():
    def __init__(self,net,Pnum,noisy=True):
        self.net=net
        self.noisy=noisy
        self.Pnum=Pnum
        self.Shortest = list(range(self.Pnum))
        self.length = self.net.length(self.Shortest)

    def start(self):
        pass


class tuihuo(Solver):
    def __init__(self, net, noisy=0, temp_dec_type=1, exchange_type=1, n=100, step=0.01):
        Solver.__init__(self, net=net, Pnum=net.points_num, noisy=noisy)
        self.exchenge_type=exchange_type
        if self.noisy>1:
            print("初始路径长度为:",self.length)
            print(self.Shortest)
        self.step=step
        self.tempreture = 1
        self.temp_dec_type=temp_dec_type    #1代表固定轮数下降，2代表固定概率下降
        if self.temp_dec_type:
            self.n=n
            self.count=0
        else:
            self.p=1/n

    def temp_change(self):
        # 1 表示温度应该降了
        if self.temp_dec_type:
            if self.count==0:
                self.count=self.n
            else:
                self.count-=1
            if self.noisy>2:
                print("当前n为%d,count为%d" %(self.n,self.count))
            return 1 if self.count==0 else 0
        else:
            p = rd.random()
            if self.noisy>2:
                print("本次生成概率为%f"%(p))
            return 1 if p<self.p else 0
    def Step(self):
        if self.exchenge_type==1:
            a, b = [rd.choice(range(self.Pnum)[1:]) for i in range(2)]
        else:
            a = rd.choice(range(self.Pnum)[1:-1])
            b = a + 1
        candidate = self.Shortest.copy()
        mid = candidate[a]
        candidate[a] = candidate[b]
        candidate[b] = mid
        del mid, a, b
        flag=-1
        ca_length = self.net.length(candidate)
        if ca_length < self.length or rd.random()<self.tempreture:
            self.Shortest = candidate
            self.length = ca_length
            flag=0
        if self.noisy + flag>1:
            print("--------------------")
            print(candidate)
            print(ca_length)
            print("当前温度为:",self.tempreture)
            print(self.Shortest)
            print(self.length)
            print("--------------------")
        if self.temp_change():
            self.tempreture -= self.step

    def end_display(self):
        print(self.Shortest)
        print(self.length)

    def start(self):
        while self.tempreture > 0:
            self.Step()
        if self.noisy > 0:
            self.end_display()
        return [self.Shortest.copy(), self.length]


class Nearst(Solver):
    def __init__(self,net_to_solve,noisy=0,start_point=0):
        Solver.__init__(self,net=net_to_solve, Pnum=net_to_solve.points_num, noisy=noisy)
        self.start_point = start_point
        self.way = None

    def start(self):
        p = {}
        for i in range(10):
            p[i]=False
        edges = self.net.edges()
        for i in range(edges.shape[0]):
            edges[i,i]=math.inf
        nearst = np.argmin(edges,axis=1)

        tmp = 0
        way = []
        while True:
            p[tmp] = True
            way.append(tmp)
            if reduce(func_and, [p[i] for i in p]):
                break
            while True:
                nxt = nearst[tmp]
                if p[nxt]:
                    edges[tmp, nxt] = math.inf
                    edges[nxt, tmp] = math.inf
                    nearst = np.argmin(edges, axis=1)
                else:
                    break
            tmp = nearst[tmp]
        self.way = tuple(way)
        return [way, self.net.length(self.way)]










