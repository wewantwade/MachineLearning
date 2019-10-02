本项目主要是用一个人工神经网络的模型实现手写数字的识别
主要注意事项有以下几点：
1、初始化链路矩阵的时候注意
        #注意括号内的self.hiddenNodes, self.inputNodes放置位置
        #写反了的话矩阵维数会对不齐，报错
        self.wi_h = np.random.rand(self.hiddenNodes, self.inputNodes) - 0.5
        self.wh_o = np.random.rand(self.outputNodes, self.hiddenNodes) - 0.5
2、注意误差值的计算，严格按照公式来
       #输出层误差
        OutputsErrors=targetsList-final_outputs#(tk-Ok)#顺序减反了模型基本没有用处了
3、再就是一定要注意矩阵计算时，维数要一一对应
     必要的时候转化成numpy形式
       