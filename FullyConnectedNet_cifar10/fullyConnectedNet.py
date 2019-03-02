# 搭建2层神经网络,全连接层
import numpy as np
from layer_utils import *


class FullyConnectedNet(object):
    '''       搭建六层神经网络：5层隐藏层+1层输入层
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
        self.num_layers = len(hidden_dim) + 1  # 神经网络层数5+1=6

        # 初始化某一隐藏层的W
        for i in range(self.num_layers):
            self.params['W' + str(i + 1)] = weight_scale * \
                np.random.randn(layer_dims[i], layer_dims[i+1])
            self.params['b'+str(i+1)] = np.zeros(layer_dims[i+1])

            if self.use_batchnorm and i < len(hidden_dim):  # 相对于隐藏层
                self.params['gamma' + str(i+1)] = np.ones(layer_dims[i+1])
                self.params['beta'+str(i+1)] = np.zeros(layer_dims[i+1])

        # 归一化
        if self.use_batchnorm:
            self.bn_configs = {}
            for i in range(self.num_layers - 1):
                self.bn_configs['W'+str(i+1)] = {'mode': 'train'}

        if self.use_dropout:
            self.dp_param = {'mode': 'train', 'P': dropout}

    # 计算模型损失函数
    def loss(self, X, y=None):
        # 有dropout时训练和测试集计算方式不同
        mode = 'test' if y is None else 'train'
        if self.use_dropout:
            self.dp_param['mode'] = mode
        if self.use_batchnorm:
            for bn in self.bn_configs:
                self.bn_configs[bn]['mode'] = mode

        caches = []  # 缓存值
        cache_dropout = []
        out = X

        # 前向传播
        # 求各神经元的输出值并存储
        # 使用dropout时，记得乘以dropout失活值
        # When using batch normalization（神经网络批归一化，即BP神经网络）,
        # you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.
        for i in range(self.num_layers - 1):  # 层数循环
            # W,b赋初值
            W = self.params['W' + str(i + 1)]
            b = self.params['b' + str(i + 1)]

            # 求输入数据经过激活函数ReLu后的值，再作为第一层输入
            if self.use_batchnorm:
                bn_param = self.bn_configs['W' + str(i+1)]
                gamma = self.params['gamma' + str(i+1)]
                beta = self.params['beta' + str(i+1)]
                out, cache = affine_batchnorm_relu_forward(
                    out, W, b, gamma, beta, bn_param
                )
            else:
                out, cache = affine_relu_forward(out, W, b)
            caches.append(cache)

            if self.use_dropout:
                out, cache = dropout_forward(out, self.dp_param)
                cache_dropout.append(cache)

        # 计算输出值
        W = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]
        out, cache = affine_forward(out, W, b)
        caches.append(cache)
        scores = out

        if y is None:
            return scores  # 如果是测试集，直接输出得分值

        # 求损失值，损失函数：softmax
        loss, dscores = softmax_loss(scores, y)
        for i in range(self.num_layers):
            W = self.params['W'+str(i+1)]
            loss += 0.5*self.reg*np.sum(W*W)  # 加入正则惩罚

        # 全连接层反向传播
        # grads[k] holds the gradients
        # for self.params[k]. Don't forget to add L2 regularization!
        # 使用批归一化时，不需要使用dropout、正则化
        grads = {}
        dout, dw, db = affine_backward(dscores, caches[-1])
        # 梯度矩阵（W,b）更新
        grads['W'+str(self.num_layers)] = dw + self.reg * \
            self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db
        for i in range(self.num_layers-1)[::-1]:  # 倒序输出,i = 4,5,4,3,2,1,0
            cache = caches[i]
            if self.use_dropout:
                dout = dropout_backward(dout, cache_dropout[i])
            if self.use_batchnorm:
                dout, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(
                    dout, cache)
            else:
                dout, dw, db = affine_relu_backward(dout, cache)
            grads['W'+str(i+1)] = dw+self.reg * self.params['W'+str(i+1)]
            grads['b'+str(i+1)] = db

        return loss, grads
