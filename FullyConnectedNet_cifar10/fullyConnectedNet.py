# 搭建2层神经网络,全连接层
import numpy as np
from layer_utils import *


class FullyConnectedNet(object):
    '''搭建六层神经网络：5层隐藏层+1层输入层
    参数W维度：W1(3072,100),W2(100,100),W3(100,100),W4(100,100),W5(100,100),W6(100,10)
    参数b维度：b1(100,),    b2(100,),   b3(100,),   b4(100,),   b5(100,),   b6(100,)
    神经元个数：每个神经元的权重都在W的一个行中，100 + 100 + 100 + 100 + 100 + 1 = 501
    参数个数：W和b相加
    '''

    def __init__(self, input_dim, hidden_dim, num_classes, weight_scale, dropout=0, reg=0.0, use_batchnorm=False):
        self.reg = reg
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0

        self.params = {}  # 参数

        # 神经网络尺寸：神经元个数和参数个数
        layer_dims = [input_dim] + hidden_dim + [num_classes]  # 神经元维度
        self.num_layers = len(hidden_dim) + 1  # 神经网络尺寸:神经元个数（输入层不算）

        # 初始化某一个神经元的W
        for i in range(self.num_layers):
            self.params['W' + str(i + 1)] = weight_scale * np.random.randn()
            pass
        pass

    def Loss():
        pass
