#!/usr/bin/python3
#@Author :
#@time   : 2019/10/2 15:39
#@file   : 神经网络.py
#Software: PyCharm

import numpy as np
import scipy.special#sigmoid函数的激活公式

# 声明一个神经网络模型类
class NeuralNetwork(object):
    # 初始化三层节点数和学习率
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate

        # 初始化输入层和隐藏层，隐藏层和输出层的链路权重矩阵
        #注意括号内的self.hiddenNodes, self.inputNodes放置位置
        #写反了的话矩阵维数会对不齐，报错
        self.wi_h = np.random.rand(self.hiddenNodes, self.inputNodes) - 0.5
        self.wh_o = np.random.rand(self.outputNodes, self.hiddenNodes) - 0.5

        #激活函数，也就是sigmoid函数
        self.activation_function=lambda x:scipy.special.expit(x)

    # 模型的训练部分,主要是反向传播进行链路权值的更新
    def train(self,inputs,targetsList):
        # 转化成numpy结构,二维张量
        inputsList = np.array(inputs, ndmin=2).T
        targetsList = np.array(targetsList, ndmin=2).T

        # 中间节点的输入
        hidden_inputs = np.dot(self.wi_h, inputsList)
        # 中间节点的输出，sigmoid激活
        hidden_outputs = self.activation_function(hidden_inputs)

        # 输出层节点的输入
        final_inputs = np.dot(self.wh_o, hidden_outputs)
        # 输出层节点的输出，用sigmoid函数激活
        final_outputs = self.activation_function(final_inputs)

        #输出层误差
        OutputsErrors=targetsList-final_outputs#(tk-Ok)#顺序减反了今天竟然准确率下降很厉害
        #隐藏层误差
        HiddenErrors=np.dot(self.wh_o.T,OutputsErrors)#注意矩阵转置

        #根据公式对链路权重更新
        self.wh_o+=self.learningRate*(np.dot(OutputsErrors*final_outputs*(1-final_outputs),np.transpose(hidden_outputs)))

        self.wi_h+=self.learningRate*(np.dot(HiddenErrors*hidden_outputs*(1-hidden_outputs),np.transpose(inputsList)))


    #节点的输入输出值计算
    def query(self,inputs):
        # print("inputs测试",inputs)
        inputs = np.array(inputs, ndmin=2).T
        # print("转职后",inputs)
        #中间节点的输入
        hidden_inputs=np.dot(self.wi_h,inputs)
        #中间节点的输出，sigmoid激活
        hidden_outputs=self.activation_function(hidden_inputs)

        #输出层节点的输入
        final_inputs=np.dot(self.wh_o,hidden_outputs)
        #输出层节点的输出，用sigmoid函数激活
        final_outputs=self.activation_function(final_inputs)

        return final_outputs



if __name__ == '__main__':


    #初始化模型参数，并调用模型训练
    inputNodes=784#(28*28个像素点)
    hiddenNodes=100#根据经验指定
    outputNodes=10#对应十个数字的概率
    learningRate=0.1

    #new一个网络
    n=NeuralNetwork(inputNodes,hiddenNodes,outputNodes,learningRate)

    #读入训练数据
    trainDataFile=open("train.csv")
    #只选100个样本训练
    trainDataList=trainDataFile.readlines()[1:]#因为第一行是标题栏，无效数据
    trainDataFile.close()

    #训练模型
    for record in trainDataList:
        #将训练数据以','分割，并提取出第一个数字，第一个数字为真实数字值不是灰度
        allValues=record.split(',')
        #设置要输入到输入层784个节点的信息
        inputs=(np.asfarray(allValues[1:]))/255*0.99+0.01
        #设置图片数字与数字的对应关系，数字是几，列表哪一个位置就设为0.99
        #用来计算与训练输出值的误差
        targets=np.zeros(outputNodes)+0.01
        targets[int(allValues[0])]=0.99
        n.train(inputs,targets)


    #模型测试
    scores=[]
    tf=open("test.csv")
    testdata=tf.readlines()
    tf.close()
    for data in testdata:
        currNum=int(data[0])
        print("当前的数字是:",currNum)
        allValues=data.split(',')
        inputs=(np.asfarray(allValues[1:]))/255*0.99+0.01#初始化输入层输入参数
        predicts=n.query(inputs)#调用模型识别
        #找到列表中最大的那个值对应的索引
        lable=np.argmax(predicts)
        print("predicts结果lable：",lable)
        if lable == currNum:
            scores.append(1)
        else:
            scores.append(0)
    print("得分",scores)
    scores_array=np.asarray(scores)
    print("模型准确率为:",scores_array.sum()/scores_array.size)

