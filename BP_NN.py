import math
import random

import matplotlib.pyplot as plt
import numpy as np

random.seed(0.5)


def Sigmoid(x):
    '''激活函数'''
    y = 1 / (1 + math.exp(-x))
    return y


class BP:
    '''四层BP神经网络'''

    def __init__(self, Ni, Nh2, Nh3, No):
        # 定义一个输入层，两个隐藏层，一个输出层的节点数
        self.Ni = Ni + 1
        self.Nh2 = Nh2
        self.Nh3 = Nh3
        self.No = No

        # 每一层的输出向量
        self.yi = np.ones(self.Ni)
        self.yh2 = np.ones(self.Nh2)
        self.yh3 = np.ones(self.Nh3)
        self.yo = np.ones(self.No)

        # 随机初始化权重矩阵
        self.w2 = np.random.rand(self.Ni, self.Nh2)
        self.w3 = np.random.rand(self.Nh2, self.Nh3)
        self.wo = np.random.rand(self.Nh3, self.No)

    #
    def update(self, input):
        # 激活输入层
        for i in range(self.Ni - 1):
            self.yi[i] = input[i]

        # 激活第二层（隐藏层）
        for j2 in range(self.Nh2):
            sum = 0.0
            for i in range(self.Ni):
                sum += self.yi[i] * self.w2[i][j2]
            self.yh2[j2] = Sigmoid(sum)

        # 激活第三层（隐藏层）
        for j3 in range(self.Nh3):
            sum = 0.0
            for j2 in range(self.Nh2):
                sum += self.yh2[j2] * self.w3[j2][j3]
            self.yh3[j3] = Sigmoid(sum)

        # 激活输出层
        for k in range(self.No):
            sum = 0.0
            for j3 in range(self.Nh3):
                sum += self.yh3[j3] * self.wo[j3][k]
            self.yo[k] = Sigmoid(sum)

        return self.yo[:]

    def backPropagation(self, target, lr):
        '''反向传播函数'''

        # 更新输出重权重
        d_o = np.zeros(self.No)
        E = 0.0
        for k in range(self.No):
            error = target[k] - self.yo[k] #输出误差
            d_o[k] = self.yo[k] * (1 - self.yo[k]) * error
            # d_o[k] = (1 - self.yo[k]**2) * error
            E += error ** 2
            for j3 in range(self.Nh3):
                D_wo = lr * d_o[k] * self.yh3[j3]
                self.wo[j3][k] = self.wo[j3][k] + D_wo
        Error = 0.5 * E

        # 更新第三层隐藏层权重
        d_h3 = np.zeros(self.Nh3)
        for j3 in range(self.Nh3):
            error = 0.0
            for k in range(self.No):
                error = error + d_o[k] * self.wo[j3][k]
            d_h3[j3] = self.yh3[j3] * (1 - self.yh3[j3]) * error
            # d_h3[j3] = (1 - self.yh3[j3]**2) * error
            for j2 in range(self.Nh2):
                D_w3 = lr * d_h3[j3] * self.yh2[j2]
                self.w3[j2][j3] = self.w3[j2][j3] + D_w3

        # 更新第二层隐藏层权重
        d_h2 = np.zeros(self.Nh2)
        for j2 in range(self.Nh2):
            error = 0.0
            for j3 in range(self.Nh3):
                error = error + d_h3[j3] * self.w3[j2][j3]
            d_h2[j2] = self.yh2[j2] * (1 - self.yh2[j2]) * error
            # d_h2[j2] = (1 - self.yh2[j2]**2) * error
            for i in range(self.Ni):
                D_w2 = lr * d_h2[j2] * self.yi[i]
                self.w2[i][j2] = self.w2[i][j2] + D_w2
        return Error

    def train(self, samples, target, iteration, lr):
        # samples:样本 target:分类目标 iteration：迭代次数 lr：学习率
        for i in range(iteration):
            error = 0.0
            for r in range(samples.shape[0]):
                input = samples[r]
                t = np.zeros(1)
                t[0] = target[r]
                self.update(input)
                error = error + self.backPropagation(t, lr)
            if i % 100 == 0:
                plt.scatter(i, error, c='r', linewidths=0.5)
                print('误差 %-.5f' % error)

    def test(self, samples):
        for r in range(samples.shape[0]):
            print(samples[r], '->', self.update(samples[r]))


if __name__ == '__main__':
    # 随机生成数据
    samples = np.zeros((20, 2))
    target = np.zeros(20)
    plt.figure()
    for i in range(10):
        while True:
            x1 = random.uniform(-5, 5)
            y1 = random.uniform(-5, 5)
            if x1 ** 2 + y1 ** 2 < 25:
                samples[i][0] = x1
                samples[i][1] = y1
                break
        while True:
            x0 = random.uniform(-10, 10)
            y0 = random.uniform(-10, 10)
            if x0 ** 2 + y0 ** 2 > 25:
                samples[i + 10][0] = x0
                samples[i + 10][1] = y0
                break

        target[i] = 1
        plt.scatter(samples[i][0], samples[i][1], c='b')
        plt.scatter(samples[i + 10][0], samples[i + 10][1], c='g')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei黑体
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Initial Data")

    # 用BP神经网络训练和输出结果
    bp = BP(2, 4, 5, 1)
    plt.figure()
    bp.train(samples, target, 10000, 1)
    plt.xlabel("iterations")
    plt.ylabel("error")
    #plt.title("Learning Rate=0.5")
    plt.title("2451")
    x_ticks = np.arange(0, 20000, 2000)
    y_ticks = np.arange(0, 5, 0.2)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    bp.test(samples)
    plt.show()
